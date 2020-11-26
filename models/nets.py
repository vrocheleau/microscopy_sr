import torch
from .ABPN import *
from .DBPN.DBPN import Net as DBPN
from .DBPN.DBPN import Net16 as DBPN16
from .FSRCNN import FSRCNN
from .RCAN import RCAN
from .EDSR import EDSR
from .DRRN import DRRN
from .MEMNET import MemNet
from utils import load_state_dict_replace


def get_abpn(channels, scale_factor):
    if scale_factor == 4:
        return ABPN_v5(channels, 32)
    elif scale_factor == 8:
        return ABPN_v5(channels, 32, kernel=10, stride=8, scale_factor=8)
    elif scale_factor == 16:
        return ABPN_v3(channels, 32)


def get_dbpn(channels, scale_factor):
    if scale_factor == 16:
        return DBPN16(num_channels=channels, base_filter=64, feat=256, num_stages=7)
    else:
        return DBPN(num_channels=channels, base_filter=64, feat=256, num_stages=7, scale_factor=scale_factor)


def get_fsrcnn(channels, scale_factor):
    return FSRCNN(scale_factor=scale_factor, num_channels=channels)

def get_rcan(channels, scale_factor):
    return RCAN(scale_factor, channels)

def get_edsr(channels, scale_factor):
    return EDSR(scale=scale_factor, n_colors=channels)

def get_drrn(channels, scale_factor):
    return DRRN(num_chanels=channels, scale_factor=scale_factor)

def get_memnet(channels, scale_factor):
    return MemNet(in_channels=channels, scale_factor=scale_factor)

def get_esrgan(channels, scale_factor):
    from .ESRGAN import GeneratorRRDB, Discriminator

    g = GeneratorRRDB(channels=channels, scale_factor=scale_factor)
    d = Discriminator(num_in_ch=channels)
    return g, d

def get_rrdb(channels, scale_factor):
    from .ESRGAN import GeneratorRRDB

    g = GeneratorRRDB(channels=channels, scale_factor=scale_factor)
    return g

models = {
    'abpn': get_abpn,
    'dbpn': get_dbpn,
    'fsrcnn': get_fsrcnn,
    'rcan': get_rcan,
    'edsr': get_edsr,
    'drrn': get_drrn,
    'memnet': get_memnet,
    'esrgan': get_esrgan,
    'rrdb': get_rrdb
}

abpn_pretrained_paths = {
    4: 'pretrained/ABPN/ABPN_4x.pth',
    8: 'pretrained/ABPN/ABPN_8x.pth',
    16: 'pretrained/ABPN/ABPN_16x.pth'
}

dbpn_pretrained_paths = {
    4: 'pretrained/DBPN/DBPN_x4.pth',
    8: 'pretrained/DBPN/DBPN_x8.pth',
}

pretrained_paths = {
    'abpn': abpn_pretrained_paths,
    'dbpn': dbpn_pretrained_paths
}


def get_net(name, chanels, scale_factor, state_dict=None, is_gan=False):

    model = models[name](chanels, scale_factor)
    if is_gan:
        model, discriminator = model
        if state_dict:
            model.load_state_dict(torch.load(state_dict))
        return model, discriminator

    if state_dict:
        model.load_state_dict(torch.load(state_dict))
    return model