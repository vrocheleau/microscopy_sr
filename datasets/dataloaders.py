from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import SrDataset
from utils import *
import os
from sacred import Ingredient

def get_datasets(splits_dir, chanels, scale_factor, patch_size, preload=False, augment=False):

    train_csv = os.path.join(splits_dir, 'train.csv')
    test_csv = os.path.join(splits_dir, 'test.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = SrDataset(train_csv, chanels, scale_factor, patch_size, trans, preload, augment)
    test_ds = SrDataset(test_csv, chanels, scale_factor, None, trans, preload, augment=False)
    val_ds = SrDataset(val_csv, chanels, scale_factor, patch_size, trans, preload, augment=False)

    return train_ds, test_ds, val_ds