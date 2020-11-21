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


def get_abpn(chanels, scale_factor):
    if scale_factor == 4:
        return ABPN_v5(chanels, 32)
    elif scale_factor == 8:
        return ABPN_v5(chanels, 32, kernel=10, stride=8, scale_factor=8)
    elif scale_factor == 16:
        return ABPN_v3(chanels, 32)


def get_dbpn(chanels, scale_factor):
    if scale_factor == 16:
        return DBPN16(num_channels=chanels, base_filter=64, feat=256, num_stages=7)
    else:
        return DBPN(num_channels=chanels, base_filter=64, feat=256, num_stages=7, scale_factor=scale_factor)


def get_fsrcnn(chanels, scale_factor):
    return FSRCNN(scale_factor=scale_factor, num_channels=chanels)

def get_rcan(chanels, scale_factor):
    return RCAN(scale_factor, chanels)

def get_edsr(chanels, scale_factor):
    return EDSR(scale=scale_factor, n_colors=chanels)

def get_drrn(chanels, scale_factor):
    return DRRN(num_chanels=chanels, scale_factor=scale_factor)

def get_memnet(chanels, scale_factor):
    return MemNet(in_channels=chanels, scale_factor=scale_factor)

models = {
    'abpn': get_abpn,
    'dbpn': get_dbpn,
    'fsrcnn': get_fsrcnn,
    'rcan': get_rcan,
    'edsr': get_edsr,
    'drrn': get_drrn,
    'memnet': get_memnet
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


def get_net(name, chanels, scale_factor, base_pretrain=False, state_dict=None):

    if base_pretrain:
        model = models[name](3, scale_factor)

        if name == 'dbpn':
            if scale_factor == 16:
                model.load_subnets_state_dicts()
            else:
                load_state_dict_replace(model, pretrained_paths[name][scale_factor], 'module.', '')
        elif name == 'abpn':
            if scale_factor == 8:
                load_state_dict_replace(model, pretrained_paths[name][scale_factor], '', '', strict=False)
            else:
                load_state_dict_replace(model, pretrained_paths[name][scale_factor], 'module.', '', strict=True)

        model.patch_input_dim(chanels)
    else:
        model = models[name](chanels, scale_factor)

        if state_dict:
            model.load_state_dict(torch.load(state_dict))

    return model