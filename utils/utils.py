from pathlib import Path
import csv
from torch.utils.data.sampler import Sampler
import torch
import copy
from torchvision import transforms
import torch.utils.data as data
import numpy as np
from collections import OrderedDict


def load_state_dict_replace(model, state_dict_path, pattern, replace_pattern, strict=True):
    print(len(model.state_dict()))
    new_state_dict = OrderedDict()
    state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
    for k, v in state_dict.items():
        name = k.replace(pattern, replace_pattern)
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=strict)
    return model

def get_files(dir_path, ext):
    return [str(path.name) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def get_paths(dir_path, ext):
    return [str(path) for path in Path(dir_path).rglob('*.{}'.format(ext))]


def parse_split_csv(csv_path):
    rows = []
    with open(csv_path, 'rb') as file:
        spamreader = csv.reader(file, delimiter=', ', quotechar='|')
        for row in spamreader:
            rows.append(row)
    return rows

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out

class ExpandedRandomSampler(Sampler):
    """Iterate multiple times over the same dataset instead of once.
    Args:
        length (int): initial length of the dataset to sample from
        multiplier (float): desired multiplier for the length of the dataset
    """

    def __init__(self, length, multiplier):
        self.length = length
        self.indices = [i for i in range(length)]
        self.total = round(self.length * multiplier)

    def __iter__(self):
        return (self.indices[i % self.length] for i in torch.randperm(self.total))

    def __len__(self):
        return self.total

class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, patch_size, scale_factor, stride, chanels):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.stride = stride
        self.height = 0
        self.width = 0
        self.chanels = chanels

    def split_img_tensor(self, img_tensor):
        # resize image and convert them into tensor
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        side = min(height, width, self.patch_size)
        delta = self.patch_size - side
        Z = torch.zeros([batch, channel, height+delta, width+delta])
        Z[:, :, delta//2:height+delta//2, delta//2:width+delta//2] = img_tensor
        batch, channel, new_height, new_width = Z.size()

        patch_box = []

        # split image into over-lapping pieces
        for i in range(0, new_height, self.stride):
            for j in range(0, new_width, self.stride):
                x = min(new_height, i + self.patch_size)
                y = min(new_width, j + self.patch_size)
                part = Z[:, :, x-self.patch_size:x, y-self.patch_size:y]

                patch_box.append(part)

        patch_tensor = torch.cat(patch_box, dim=0)
        return patch_tensor

    def merge_img_tensor(self, list_img_tensor):
        img_tensors = copy.copy(list_img_tensor)

        patch_size = self.patch_size * self.scale_factor
        stride = self.stride * self.scale_factor
        height = self.height * self.scale_factor
        width = self.width * self.scale_factor
        side = min(height, width, patch_size)
        delta = patch_size - side
        new_height = delta + height
        new_width = delta + width
        out = torch.zeros((1, self.chanels, new_height, new_width))
        mask = torch.zeros((1, self.chanels, new_height, new_width))

        for i in range(0, new_height, stride):
            for j in range(0, new_width, stride):
                x = min(new_height, i + patch_size)
                y = min(new_width, j + patch_size)
                mask_patch = torch.zeros((1, self.chanels, new_height, new_width))
                out_patch = torch.zeros((1, self.chanels, new_height, new_width))
                mask_patch[:, :, (x - patch_size):x, (y - patch_size):y] = 1.0
                out_patch[:, :, (x - patch_size):x, (y - patch_size):y] = img_tensors.pop(0)
                mask = mask + mask_patch
                out = out + out_patch

        out = out / mask

        out = out[:, :, delta//2:new_height - delta//2, delta//2:new_width - delta//2]

        return out


def chop_forward(img, network, patch_size, scale_factor, stride, chanels, transform=None):
    img_splitter = ImageSplitter(patch_size, scale_factor, stride, chanels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if transform:
        img = transform(img)[0]
    # channel_swap = (1, 2, 0)

    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    elif len(img.shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)



    img_patch = img_splitter.split_img_tensor(img)

    testset = data.TensorDataset(img_patch)
    test_dataloader = data.DataLoader(testset, num_workers=6,
                                       drop_last=False, batch_size=1, shuffle=False)
    out_box = []

    for iteration, batch in enumerate(test_dataloader, 1):
        input = torch.autograd.Variable(batch[0]).to(device)
        with torch.no_grad():
            prediction = network(input)

        torch.cuda.synchronize()

        for j in range(prediction.shape[0]):
            out_box.append(prediction[j, :, :, :])

    SR = img_splitter.merge_img_tensor(out_box)
    SR = SR.data.numpy()
    print(SR.shape)
    return SR


def patchify_forward(img, model, patch_size, scaling_factor, overlap=10, transform=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b, c, h, w = img.shape

    if transform:
        img = transform(img)[0]

    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    elif len(img.shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    patches = patchify_tensor(img, patch_size, overlap)

    testset = data.TensorDataset(patches)
    test_dataloader = data.DataLoader(testset, num_workers=6,
                                      drop_last=False, batch_size=4, shuffle=False)
    out_box = []

    for index, sample in enumerate(test_dataloader):
        input = torch.autograd.Variable(sample[0]).to(device)
        with torch.no_grad():
            prediction = model(input)
            out_box.append(prediction)

    sr_patches = torch.cat(out_box, 0)

    SR = recompose_tensor(sr_patches, full_height=h*scaling_factor, full_width=w*scaling_factor, overlap=overlap*scaling_factor)
    SR = min_max_scaler(SR)
    return SR

def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
    return torch.cat(patches, 0)


def recompose_tensor(patches, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor


def min_max_scaler(tensor):
    return (tensor - tensor.min())/(tensor.max() - tensor.min())