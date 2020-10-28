import numpy as np
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.nets import get_net
from models.ABPN import *
from datasets.dataloaders import get_datasets
import warnings
from math import log10
from piq import psnr
from piq import ssim
from utils import patchify_forward, min_max_scaler
from torchvision.utils import save_image
import os
import json
from matplotlib import pyplot as plt
from aicsimageio import AICSImage, imread
import torch.utils.data as utils
import torchvision.transforms as transforms
from torchvision.utils import save_image

abpn_pretrained_paths = {
    4: 'checkpoints/trained/ABPN_4x.pth',
    8: 'checkpoints/trained/ABPN_8x.pth',
    16: 'checkpoints/trained/ABPN_16x.pth'
}

dbpn_pretrained_paths = {
    4: 'checkpoints/trained/DBPN_4x.pth',
    8: 'checkpoints/trained/DBPN_8x.pth',
    16: 'checkpoints/trained/DBPN_16x.pth',
}


pretrained_paths = {
    'abpn': abpn_pretrained_paths,
    'dbpn': dbpn_pretrained_paths
}

to_tensor = transforms.ToTensor()

def eval(model, czi_file, patch_size, scale_factor, save_dir, chanel):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    stack = AICSImage(czi_file).get_image_data()
    empty_axis = stack.shape.index(1)
    stack = stack.squeeze(empty_axis)

    subjects, chanels, time_axis, height, width = stack.shape
    for i in tqdm(range(subjects)):
        sub_stack = stack[i][chanel]
        print(sub_stack.shape)

        T, _, _ = sub_stack.shape

        sub_dir = os.path.join(save_dir, 'serie_{}'.format(i))
        os.makedirs(sub_dir, exist_ok=True)

        out = []
        for t in range(T):
            input = to_tensor(sub_stack[t]).unsqueeze(0)
            result = patchify_forward(input, model, patch_size, scale_factor)
            print(result.shape)
            save_image(input.squeeze(0), os.path.join(sub_dir, 'in_t{}_.png'.format(t)))
            save_image(result.squeeze(0), os.path.join(sub_dir, 'out_t{}_.png'.format(t)))


# Shape [31, 1, 3, 5, 512, 512]
# [subject, na, chanels, time_axis, height, width]
if __name__ == '__main__':

    name = 'dbpn'  # abpn or dbpn

    # path = '/home/victor/datasets/microscopy_luke/Movies_SuperRes/Movies_SuperRes/LifeAct Live Imaging/High res end of ife imaging (72 hr, scan speed4).czi'
    path = '/home/victor/datasets/microscopy_luke/Movies_SuperRes/Movies_SuperRes/03March2020_4day_2020_03_03__19_59_15(14).czi'

    file_name = path.split('/')[-1].replace('.czi', '')

    scale_factor = 4
    chanel = 2
    pretrained = True
    multi_gpu = True
    interp = False

    save_dir = 'films_out/{}/chanel_{}/{}/{}x'.format(file_name, chanel, name.upper(), scale_factor)
    os.makedirs(save_dir, exist_ok=True)

    model = get_net(name=name, chanels=1, scale_factor=scale_factor,
                    base_pretrain=False, state_dict=pretrained_paths[name][scale_factor])

    eval(model, path, 64, scale_factor, save_dir, chanel)