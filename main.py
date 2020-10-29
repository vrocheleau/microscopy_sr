from sacred import Experiment
import numpy as np
import torch
import torch.nn as nn
from utils import ExpandedRandomSampler
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
import time
from train import train
from test import test
from sacred.observers import FileStorageObserver, MongoObserver

warnings.filterwarnings("ignore", module=".*aicsimageio")

ex = Experiment()
ex.observers.append(FileStorageObserver('sacred_runs'))
# ex.observers.append(MongoObserver())
# ex.observers.append(MongoObserver(
#     url='mongodb://mongo_user:mongo_password@localhost:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false',
#     db_name='sacred'))

def opt_sch_ABPN(model):
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = None
    return optimizer, scheduler

@ex.config
def conf():
    name = 'abpn' # abpn, dbpn, fsrcnn
    scale_factor = 4
    pretrained = False
    multi_gpu = True
    batch_size = 32
    chanels = [2]
    epochs = 100
    preload = True
    patch_size = 32

@ex.automain
def main(name, scale_factor, pretrained, multi_gpu, batch_size, patch_size, chanels, epochs, preload):

    # Train phase
    train_ds, test_ds, val_ds = get_datasets('datasets/splits/czi',
                                             chanels=chanels, scale_factor=scale_factor, patch_size=patch_size * scale_factor,
                                             preload=preload, augment=True)

    model = get_net(name=name, chanels=1, scale_factor=scale_factor, base_pretrain=pretrained)

    if multi_gpu:
        model = nn.DataParallel(model)

    save_path = 'checkpoints/{}'.format(name.upper())
    save_name = os.path.join(save_path, '{}_{}x.pth'.format(name.upper(), scale_factor))
    os.makedirs(save_path, exist_ok=True)

    model = train(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size, opt_sch_callable=opt_sch_ABPN, loss_object=nn.L1Loss(),
          checkpoint_path=save_name, ex=ex)

    # Test phase
    save_dir = 'out/{}/{}x'.format(name.upper(), scale_factor)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'imgs'), exist_ok=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)

    metrics = test(model, test_loader, patch_size=patch_size, scale_factor=scale_factor, test=True, save_dir=save_dir)

    mean_loss = np.asarray(metrics['losses']).mean()
    mean_ssim = np.asarray(metrics['ssim']).mean()
    mean_psnr = np.asarray(metrics['psnr']).mean()

    print("Mean loss: ", metrics['losses'].mean())
    print("Mean ssim: ", metrics['ssim'].mean())
    print("Mean psnr: ", metrics['psnr'].mean())

    ex.info['mean_loss'] = mean_loss
    ex.info['mean_ssim'] = mean_ssim
    ex.info['mean_psnr'] = mean_psnr
