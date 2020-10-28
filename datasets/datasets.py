from PIL import Image
import torch as torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from random import random, randint
from torchvision import transforms
from utils import *
from aicsimageio import AICSImage
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

class SrDataset(Dataset):

    def __init__(self, csv_path, chanels, scale_factor, patch_size, transform, preload=False, augment=False):

        self.rows = csv_reader(csv_path)
        self.chanels = chanels
        self.scale_factor = scale_factor
        self.transform = transform
        self.preload = preload
        self.augment = augment
        self.patch_size = patch_size
        self.n = len(self.rows)
        self.images = {}

        if preload:
            self.images = self.load_images(self.rows)

    def __getitem__(self, item):

        img_path = self.rows[item][0]

        if self.preload or item in self.images.keys():
            image = self.images[item]
        else:
            image = self.load_image(img_path)
            self.images[item] = image

        if self.augment:

            # random rotate
            angle = randint(0, 3) * 90
            image = image.rotate(angle)

            # random crop
            if self.patch_size is not None:
                rand_crop = transforms.RandomCrop(size=self.patch_size)
                image = rand_crop(image)

            # flip
            if random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.patch_size is not None:
            rand_crop = transforms.CenterCrop(size=self.patch_size)
            image = rand_crop(image)

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        image = image.unsqueeze(0)
        lr = F.interpolate(image, scale_factor=1/self.scale_factor, mode='bicubic', align_corners=True)

        return image.squeeze(0), lr.squeeze(0)

    def __len__(self):
        return len(self.rows)

    def load_images(self, rows):
        images = {}
        for i, f in enumerate(tqdm(rows, 'pre-loading dataset images')):
            images[i] = self.load_image(f[0])
        return images

    def load_image(self, path):
        img = AICSImage(path)
        img = img.get_image_data("CZYX", S=0, T=0, dim=0)
        img = img.squeeze(1)[self.chanels]
        img = img.transpose(1, 2, 0)
        img = to_pil_image(img)
        return img
