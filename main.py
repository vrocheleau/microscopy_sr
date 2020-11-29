from sacred import Experiment
import numpy as np
import torch
from torch.optim import lr_scheduler, Adam, SGD
from torch.utils.data import DataLoader
from models.ABPN import *
from models.nets import get_net
from datasets.dataloaders import get_datasets
import os
import warnings
from train import train, train_gan
from test import test
from sacred.observers import FileStorageObserver
from losses.esrgan import PerceptualLoss

warnings.filterwarnings("ignore", module=".*aicsimageio")

ex = Experiment()

@ex.config
def conf():
    name = 'rcan'
    scale_factor = 4
    fold = 0
    multi_gpu = False
    gpu_id = 0
    batch_size = 16
    chanels = [2]
    lr = 1e-4
    epochs = 2000
    val_intervals = 5
    preload = True
    patch_size = 32
    lr_step = 1000
    is_gan = False
    psnr_oriented = False
    state_dict = None
    seed = 0

@ex.capture
def opt_sch_ABPN(model, lr_step, lr):
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step)
    return optimizer, scheduler

@ex.capture
def g_opt_sch(model, lr_step, lr):
    optimizer_generator = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_generator = lr_scheduler.StepLR(optimizer_generator, step_size=lr_step)
    return optimizer_generator, scheduler_generator

@ex.capture
def d_opt_sch(model, lr_step, lr):
    # optimizer_generator = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer_generator = SGD(model.parameters(), lr=lr)
    scheduler_generator = lr_scheduler.StepLR(optimizer_generator, step_size=lr_step)
    return optimizer_generator, scheduler_generator

@ex.capture
def losses(device, name):
    if name == 'esrgan':
        adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
        content_criterion = nn.L1Loss().to(device)
        perception_criterion = PerceptualLoss().to(device)
        return adversarial_criterion, content_criterion, perception_criterion
    else:
        return nn.L1Loss().to(device)

@ex.capture
def get_exp_id(_run):
    return _run._id, _run.observers[0].basedir


@ex.automain
def main(name, scale_factor, multi_gpu, batch_size, patch_size, chanels, epochs, val_intervals, preload, is_gan, psnr_oriented, state_dict, fold, gpu_id):

    torch.backends.cudnn.benchmark = True

    exp_id, basedir = get_exp_id()

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'

    # Train phase
    train_ds, test_ds, val_ds = get_datasets(splits_dir='datasets/splits', fold_num=fold,
                                             chanels=chanels, scale_factor=scale_factor, patch_size=patch_size * scale_factor,
                                             preload=preload, augment=True)

    save_path = 'checkpoints/{}'.format(name.upper())
    save_name = os.path.join(save_path, '{}_{}x.pth'.format(name.upper(), scale_factor))
    os.makedirs(save_path, exist_ok=True)

    if is_gan:
        generator, discriminator = get_net(name=name, chanels=len(chanels), scale_factor=scale_factor, is_gan=is_gan, state_dict=state_dict)
        if multi_gpu:
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)

        model = train_gan(generator=generator,
                          discriminator=discriminator,
                          train_ds=train_ds,
                          val_ds=val_ds,
                          batch_size=batch_size,
                          g_opt_sch=g_opt_sch,
                          d_opt_sch=d_opt_sch,
                          losses=losses,
                          val_intervals=val_intervals,
                          device=device,
                          checkpoint_path=save_name, ex=ex, epochs=epochs, psnr_oriented=psnr_oriented)

    else:
        model = get_net(name=name,
                        chanels=len(chanels),
                        scale_factor=scale_factor,
                        state_dict=state_dict)
        if multi_gpu:
            model = nn.DataParallel(model)
        model = train(model, train_ds, val_ds, epochs=epochs, val_intervals=val_intervals, batch_size=batch_size,
                      opt_sch_callable=opt_sch_ABPN, loss_object=nn.L1Loss(), device=device, checkpoint_path=save_name, ex=ex)

    # Save state dict in sacred exp dir
    if isinstance(model, nn.DataParallel):
        best_state_dict = model.module.state_dict()
    else:
        best_state_dict = model.state_dict()

    exp_dir = os.path.join(basedir, exp_id)
    torch.save(best_state_dict, os.path.join(exp_dir, '{}_{}x.pth'.format(name.upper(), scale_factor)))

    # Test phase
    save_dir = 'out/{}/{}x'.format(name.upper(), scale_factor)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'imgs'), exist_ok=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)

    metrics = test(model, test_loader, patch_size=patch_size, scale_factor=scale_factor, device=device, test=True, save_dir=save_dir)

    mean_loss = np.asarray(metrics['losses']).mean()
    mean_ssim = np.asarray(metrics['ssim']).mean()
    mean_psnr = np.asarray(metrics['psnr']).mean()

    print("Mean loss: ", metrics['losses'].mean())
    print("Mean ssim: ", metrics['ssim'].mean())
    print("Mean psnr: ", metrics['psnr'].mean())

    ex.info['mean_loss'] = mean_loss
    ex.info['mean_ssim'] = mean_ssim
    ex.info['mean_psnr'] = mean_psnr
