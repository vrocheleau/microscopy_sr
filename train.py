import numpy as np
import torch
import torch.nn as nn
from utils import ExpandedRandomSampler, MetricsLogger
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ABPN import *
from models.nets import get_net
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloaders import get_datasets
import os
import warnings
from sklearn import metrics

warnings.filterwarnings("ignore", module=".*aicsimageio")


def train_gan(generator: nn.Module, discriminator: nn.Module, train_ds, val_ds, epochs, val_intervals, batch_size,
             g_opt_sch, d_opt_sch, losses, psnr_oriented, device, checkpoint_path=None, ex=None):

    generator.to(device)
    discriminator.to(device)

    adversarial_criterion, content_criterion, perception_criterion = losses(device)

    adversarial_criterion.to(device)
    content_criterion.to(device)
    perception_criterion.to(device)

    g_optimizer, g_scheduler = g_opt_sch(generator)
    d_optimizer, d_scheduler = d_opt_sch(discriminator)

    if psnr_oriented:
        content_loss_factor = 1
        perceptual_loss_factor = 0
        adversarial_loss_factor = 0
    else:
        content_loss_factor = 1e-2
        perceptual_loss_factor = 1
        adversarial_loss_factor = 5e-3

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=ExpandedRandomSampler(len(train_ds), multiplier=1),
                              num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_val_loss = float('inf')

    if isinstance(generator, torch.nn.DataParallel):
        best_state_dict = deepcopy(generator.module.state_dict())
    else:
        best_state_dict = deepcopy(generator.state_dict())

    writer = SummaryWriter()

    mini_batch = 0

    real_acc_logger = MetricsLogger(evaluation=metrics.accuracy_score)
    fake_acc_logger = MetricsLogger(evaluation=metrics.accuracy_score)
    g_loss_logger = MetricsLogger()
    d_loss_logger = MetricsLogger()

    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        real_acc_logger.reset()
        fake_acc_logger.reset()
        g_loss_logger.reset()
        d_loss_logger.reset()

        for HR, LR in tqdm(train_loader, ncols=100, desc='[{}/{}]Training'.format(epoch, epochs)):
            HR, LR = HR.to(device), LR.to(device)

            real_labels = torch.ones(HR.size(0)).to(device)
            fake_labels = torch.zeros(HR.size(0)).to(device)

            # Train generator
            g_optimizer.zero_grad()

            fake_HR = generator(LR)

            score_real = discriminator(HR)
            score_fake = discriminator(fake_HR)

            real_acc = real_acc_logger.add_batch(real_labels.cpu().numpy(), (score_real.detach().cpu().numpy() >= 0.5) * 1)
            fake_acc = fake_acc_logger.add_batch(fake_labels.cpu().numpy(), (score_fake.detach().cpu().numpy() < 0.5) * 1)

            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
            adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            perceptual_loss = perception_criterion(HR, fake_HR)
            content_loss = content_criterion(fake_HR, HR)

            generator_loss = adversarial_loss * adversarial_loss_factor + \
                             perceptual_loss * perceptual_loss_factor + \
                             content_loss * content_loss_factor

            generator_loss.backward()

            g_optimizer.step()

            # Train discriminator
            d_optimizer.zero_grad()

            score_real = discriminator(HR)
            score_fake = discriminator(fake_HR.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()

            adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
            adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)

            discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            discriminator_loss.backward()

            d_optimizer.step()

            g_scheduler.step()
            d_scheduler.step()

            g_loss_logger.add_value(generator_loss.item())
            d_loss_logger.add_value(discriminator_loss.item())

            writer.add_scalar('Loss_mini/train_g', generator_loss.item(), mini_batch)
            writer.add_scalar('Loss_mini/train_d', discriminator_loss.item(), mini_batch)
            writer.add_scalar('Acc_mini/fake', fake_acc, mini_batch)
            writer.add_scalar('Acc_mini/real', real_acc, mini_batch)

            mini_batch += 1

        writer.add_scalar('Loss/train_g', g_loss_logger.value(), epoch)
        writer.add_scalar('Loss/train_d', d_loss_logger.value(), epoch)
        writer.add_scalar('Acc/fake', fake_acc_logger.value(), epoch)
        writer.add_scalar('Acc/real', real_acc_logger.value(), epoch)

        # Eval model
        if epoch != 0 and epoch % val_intervals == 0:
            val_metrics = eval(generator, val_loader, nn.L1Loss(), test=False)

            if val_metrics['losses'].mean() <= best_val_loss:
                best_val_loss = val_metrics['losses'].mean()
                if isinstance(generator, torch.nn.DataParallel):
                    best_state_dict = deepcopy(generator.module.state_dict())
                else:
                    best_state_dict = deepcopy(generator.state_dict())

                if checkpoint_path is not None:
                    best_path = checkpoint_path.replace('.pth', '_best.pth')
                    print("Checkpoint saving state dict")
                    torch.save(best_state_dict, best_path)

            print('Val loss: {}'.format(val_metrics['losses'].mean()))
            writer.add_scalar('Loss/val_g', val_metrics['losses'].mean(), epoch)

            if ex is not None:
                ex.log_scalar('val_loss', val_metrics['losses'].mean())

        if isinstance(generator, torch.nn.DataParallel):
            torch.save(deepcopy(generator.module.state_dict()), checkpoint_path)
        else:
            torch.save(deepcopy(generator.state_dict()), checkpoint_path)

    return generator

def train(model, train_ds, val_ds, epochs, val_intervals,batch_size, opt_sch_callable, loss_object, device, checkpoint_path=None, ex=None):

    model.to(device)

    optimizer, scheduler = opt_sch_callable(model)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=ExpandedRandomSampler(len(train_ds), multiplier=1),
                              num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_val_loss = float('inf')
    if isinstance(model, torch.nn.DataParallel):
        best_state_dict = deepcopy(model.module.state_dict())
    else:
        best_state_dict = deepcopy(model.state_dict())

    writer = SummaryWriter()

    mini_batch = 0

    for epoch in range(epochs):

        model.train()


        for HR, LR in tqdm(train_loader, ncols=100, desc='[{}/{}]Training'.format(epoch, epochs)):
            HR, LR = HR.to(device), LR.to(device)

            out = model(LR)
            loss = loss_object(out, HR)

            optimizer.zero_grad()
            loss.backward()
            writer.add_scalar('Loss_mini/train_g', loss.item(), mini_batch)

            optimizer.step()

            mini_batch += 1

        if scheduler is not None:
            scheduler.step()

        # Eval model
        if epoch != 0 and epoch % val_intervals == 0:
            val_metrics = eval(model, val_loader, loss_object, test=False, device=device)
            if val_metrics['losses'].mean() <= best_val_loss:
                best_val_loss = val_metrics['losses'].mean()
                if isinstance(model, torch.nn.DataParallel):
                    best_state_dict = deepcopy(model.module.state_dict())
                else:
                    best_state_dict = deepcopy(model.state_dict())
                if checkpoint_path is not None:
                    print("Checkpoint saving state dict")
                    torch.save(best_state_dict, checkpoint_path)

            print('Val loss: {}'.format(val_metrics['losses'].mean()))
            writer.add_scalar('Loss/val', val_metrics['losses'].mean(), epoch)

            if ex is not None:
                ex.log_scalar('val_loss', val_metrics['losses'].mean())

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(best_state_dict)
    else:
        model.load_state_dict(best_state_dict)

    return model


def eval(model, loader, loss_object, device, test=False):
    model.eval()

    all_losses = []

    pbar = tqdm(loader, ncols=80, desc='Test' if test else 'Validation')
    with torch.no_grad():
        for HR, LR in pbar:
            HR, LR = HR.to(device), LR.to(device, non_blocking=True)

            logits = model(LR)
            loss = loss_object(logits, HR).item()

            all_losses.append(loss)

        all_losses = np.array(all_losses)

    metrics = {
        'losses': all_losses,
    }

    return metrics

def opt_sch_ABPN(model):
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = None
    return optimizer, scheduler
