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

warnings.filterwarnings("ignore", module=".*aicsimageio")

def save_fig(lr, hr, sr, sr_bic, sr_metrics, bic_metrics, factor, file_name):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={'wspace': 0})
    fig.suptitle('{}x upsampling'.format(factor))

    lr = lr.cpu().numpy().squeeze(0).squeeze(0)
    hr = hr.cpu().numpy().squeeze(0).squeeze(0)
    sr = sr.cpu().numpy().squeeze(0).squeeze(0)
    sr_bic = sr_bic.cpu().numpy().squeeze(0).squeeze(0)

    ax1.imshow(hr, cmap='gray')
    ax1.set_title('Ground truth HR')

    ax2.imshow(lr, cmap='gray')
    ax2.set_title('Bicubic LR')

    ax3.imshow(sr, cmap='gray')
    ax3.set_title('predicted SR \n (SSIM {}, PSNR {})'.format(
        round(sr_metrics[0], 3),
        round(sr_metrics[1], 3)))

    ax4.imshow(sr_bic, cmap='gray')
    ax4.set_title('bicubic SR \n (SSIM {}, PSNR {})'.format(
        round(bic_metrics[0], 3),
        round(bic_metrics[1], 3)))

    fig.savefig(file_name, dpi=300)
    plt.close(fig)

def test(model, loader, patch_size, scale_factor, save_dir, test=False, interp=False):
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_object = nn.L1Loss()

    all_losses = []
    all_psnr = []
    all_ssim = []

    pbar = tqdm(loader, ncols=80, desc='Test' if test else 'Validation')
    with torch.no_grad():
        for i, (HR, LR) in enumerate(pbar):
            HR, LR = HR.to(device), LR

            if interp:
                LR = LR.to(device)
                pred = F.interpolate(LR, scale_factor=scale_factor, align_corners=True, mode='bicubic')
                pred = min_max_scaler(pred)
                psnr_val = psnr(pred, HR).item()
                ssim_val = ssim(pred, HR).item()
            else:
                pred = patchify_forward(img=LR, model=model, patch_size=patch_size, scaling_factor=scale_factor)

                LR = LR.to(device)
                pred_bic = F.interpolate(LR, scale_factor=scale_factor, align_corners=True, mode='bicubic')
                pred_bic = min_max_scaler(pred_bic)

                psnr_val = psnr(pred, HR).item()
                ssim_val = ssim(pred, HR).item()
                psnr_val_bic = psnr(pred_bic, HR).item()
                ssim_val_bic = ssim(pred_bic, HR).item()

                save_fig(LR, HR, pred, pred_bic, (ssim_val, psnr_val), (ssim_val_bic, psnr_val_bic),
                         factor=scale_factor, file_name=os.path.join(save_dir, 'imgs/sample_{}.png'.format(i)))


            loss = loss_object(pred, HR).item()

            # psnr_val = psnr(pred, HR).item()
            # ssim_val = ssim(pred, HR).item()

            all_losses.append(loss)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            # save_image(pred[0], os.path.join(save_dir, 'imgs/sample_{}.png'.format(i)))

        all_losses = np.array(all_losses)
        all_psnr = np.array(all_psnr)
        all_ssim = np.array(all_ssim)


    metrics = {
        'losses': all_losses,
        'ssim': all_ssim,
        'psnr': all_psnr
    }

    return metrics

models_rgb = {
    4: ABPN_v5(3, 32),
    8: ABPN_v5(3, 32, kernel=10, stride=8, scale_factor=8),
    16: ABPN_v3(3, 32)
}
models_gs = {
    4: ABPN_v5(1, 32),
    8: ABPN_v5(1, 32, kernel=10, stride=8, scale_factor=8),
    16: ABPN_v3(1, 32)
}

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

def get_psnr(pred, target):
    mse_loss = nn.MSELoss()
    mse = mse_loss(pred, target)
    psnr = 10 * log10(pow(255.0, 2) / mse)
    return psnr


if __name__ == '__main__':

    name = 'dbpn' # abpn or dbpn
    scale_factor = 4
    pretrained = True
    multi_gpu = True
    interp = False

    save_dir = 'out/{}/{}x'.format(name.upper(),scale_factor)
    if interp:
        save_dir = save_dir + '_bi'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'imgs'), exist_ok=True)

    # pretrained_path = pretrained_paths[scale_factor]

    train_ds, test_ds, val_ds = get_datasets('datasets/splits/czi',
                                             chanels=[2], scale_factor=scale_factor,
                                             patch_size=None, preload=False, augment=True)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=6)

    model = get_net(name=name, chanels=1, scale_factor=scale_factor,
                    base_pretrain=False, state_dict=pretrained_paths[name][scale_factor])

    if multi_gpu:
        model = nn.DataParallel(model)

    metrics = test(model, test_loader, patch_size=32, scale_factor=scale_factor, test=True, interp=interp, save_dir=save_dir)

    mean_loss = np.asarray(metrics['losses']).mean()
    mean_ssim = np.asarray(metrics['ssim']).mean()
    mean_psnr = np.asarray(metrics['psnr']).mean()

    print("Mean loss: ", metrics['losses'].mean())
    print("Mean ssim: ", metrics['ssim'].mean())
    print("Mean psnr: ", metrics['psnr'].mean())

    mean_metrics = {
        'mean_loss': mean_loss,
        'mean_ssim': mean_ssim,
        'mean_psnr': mean_psnr
    }

    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as file:
        file.write(json.dumps(mean_metrics))
