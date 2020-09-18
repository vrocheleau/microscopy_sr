import numpy as np
import torch.nn as nn
from utils import ExpandedRandomSampler
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.ABPN import *
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloaders import get_datasets

def train(model, train_ds, val_ds, epochs, batch_size, opt_sch_callable, loss_object, model_checkpoint=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer, scheduler = opt_sch_callable(model)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=ExpandedRandomSampler(len(train_ds), multiplier=1), num_workers=6)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=6)

    best_val_loss = float('inf')
    best_state_dict = deepcopy(model.state_dict())

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

        if val_metrics['losses'].mean() <= best_val_loss and model_checkpoint:
            best_val_loss = val_metrics['losses'].mean()
            best_state_dict = deepcopy(model.state_dict())


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

    train_ds, test_ds, val_ds = get_datasets('/home/victor/PycharmProjects/microscopy_sr/datasets/splits/czi',
                                             [2], 4, 160, preload=False, augment=True)

    model = ABPN_v5(1, 32)

    train(model, train_ds, val_ds, 50, 8, opt_sch_callable=opt_sch_ABPN, loss_object=nn.L1Loss())
