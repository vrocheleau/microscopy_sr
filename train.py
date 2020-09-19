import numpy as np
import torch
import torch.nn as nn
from utils import ExpandedRandomSampler
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ABPN import *
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloaders import get_datasets
import os

def train(model, train_ds, val_ds, epochs, batch_size, opt_sch_callable, loss_object, checkpoint_path=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer, scheduler = opt_sch_callable(model)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=ExpandedRandomSampler(len(train_ds), multiplier=2), num_workers=8)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=6)

    best_val_loss = float('inf')
    best_state_dict = deepcopy(model.module.state_dict())

    writer = SummaryWriter()

    for epoch in range(epochs):
        model.train()

        print('Epoch {} / {}'.format(epoch + 1, epochs))

        for HR, LR in tqdm(train_loader, ncols=100, desc='[{}/{}]Training'.format(epoch, epochs)):
            HR, LR = HR.to(device), LR.to(device, non_blocking=True)

            out = model(LR)

            loss = loss_object(out, HR)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Eval model
        val_metrics = eval(model, val_loader, loss_object, test=False)

        if val_metrics['losses'].mean() <= best_val_loss:
            best_val_loss = val_metrics['losses'].mean()
            best_state_dict = deepcopy(model.module.state_dict())

        if checkpoint_path is not None:
            torch.save(best_state_dict, checkpoint_path)

        print('Val loss: {}'.format(val_metrics['losses'].mean()))
        writer.add_scalar('Loss/val', val_metrics['losses'].mean(), epoch)

    model.load_state_dict(best_state_dict)

def eval(model, loader, loss_object, test=False):
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

if __name__ == '__main__':

    pretrained = True
    multi_gpu = True

    pretrained_path = 'pretrained/ABPN/ABPN_4x.pth'

    train_ds, test_ds, val_ds = get_datasets('datasets/splits/czi',
                                             chanels=[2], scale_factor=4, patch_size=160, preload=False, augment=True)

    if pretrained:
        model = ABPN_v5(3, 32)
        model.load_state_dict(torch.load((pretrained_path)))
        model.patch_input_dim(1, 32)
    else:
        model = ABPN_v5(1, 32)

    if multi_gpu:
        model = nn.DataParallel(model)

    save_path = 'checkpoints/ABPN/{}'.format('ABPN_4x.pth')
    os.makedirs(save_path, exist_ok=True)

    train(model, train_ds, val_ds, epochs=200, batch_size=32, opt_sch_callable=opt_sch_ABPN, loss_object=nn.L1Loss(), checkpoint_path=save_path)
