import numpy as np
from utils import  ImageSplitter
from models.ABPN import *
from datasets.dataloaders import get_datasets
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as utils
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

models_gs = {
    4: ABPN_v5(1, 32),
    8: ABPN_v5(1, 32, kernel=10, stride=8, scale_factor=8),
    16: ABPN_v3(1, 32)
}

patch_size = 128
stride = patch_size
scale_factor = 4

img_splitter = ImageSplitter(patch_size, scale_factor, stride)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def chop_forward(img, network, start, end):

    channel_swap = (1, 2, 0)
    run_time = 0
    img = transform(img)
    img = img[0].unsqueeze(0).unsqueeze(0)

    img_patch = img_splitter.split_img_tensor(img)

    testset = utils.TensorDataset(img_patch)
    test_dataloader = utils.DataLoader(testset, num_workers=6,
                                       drop_last=False, batch_size=1, shuffle=False)
    out_box = []

    for iteration, batch in enumerate(test_dataloader, 1):
        input = torch.autograd.Variable(batch[0]).to(device)

        start.record()
        with torch.no_grad():
            prediction = network(input)
        end.record()
        torch.cuda.synchronize()
        run_time += start.elapsed_time(end)

        for j in range(prediction.shape[0]):
            out_box.append(prediction[j,:,:,:])

    SR = img_splitter.merge_img_tensor(out_box)
    SR = SR.data[0].numpy().transpose(channel_swap)

    return SR, run_time

if __name__ == '__main__':

    pretrained = True
    multi_gpu = True
    lr_dir = 'LR_demo/{}x/'.format(scale_factor)
    pretrained_path = 'pretrained/ABPN/luna/ABPN_{}x.pth'.format(scale_factor)
    sr_dir = 'SR_demo/{}x'.format(scale_factor)
    os.makedirs(sr_dir, exist_ok=True)

    model = models_gs[scale_factor]
    model.load_state_dict(torch.load((pretrained_path)), strict=False)

    if multi_gpu:
        model = nn.DataParallel(model)

    LR_filename = lr_dir
    LR_image = [os.path.join(LR_filename, x) for x in os.listdir(LR_filename)]
    SR_image = [os.path.join(sr_dir, x) for x in os.listdir(LR_filename)]

    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in tqdm(range(LR_image.__len__()), ncols=100, desc="Upscaling whole images"):

        idx += 1
        img_name, ext = os.path.splitext(LR_image[i])

        LR = Image.open(LR_image[i]).convert('RGB')
        LR_90 = LR.transpose(Image.ROTATE_90)
        LR_180 = LR.transpose(Image.ROTATE_180)
        LR_270 = LR.transpose(Image.ROTATE_270)
        LR_f = LR.transpose(Image.FLIP_LEFT_RIGHT)
        LR_90f = LR_90.transpose(Image.FLIP_LEFT_RIGHT)
        LR_180f = LR_180.transpose(Image.FLIP_LEFT_RIGHT)
        LR_270f = LR_270.transpose(Image.FLIP_LEFT_RIGHT)

        with torch.no_grad():
            pred, time = chop_forward(LR, model, start, end)
            pred_90, time_90 = chop_forward(LR_90, model, start, end)
            pred_180, time_180 = chop_forward(LR_180, model, start, end)
            pred_270, time_270 = chop_forward(LR_270, model, start, end)
            pred_f, time_f = chop_forward(LR_f, model, start, end)
            pred_90f, time_90f = chop_forward(LR_90f, model, start, end)
            pred_180f, time_180f = chop_forward(LR_180f, model, start, end)
            pred_270f, time_270f = chop_forward(LR_270f, model, start, end)

        compute_time = time + time_90 + time_180 + time_270 + time_f + time_90f + time_180f + time_270f
        pred_90 = np.rot90(pred_90, 3)
        pred_180 = np.rot90(pred_180, 2)
        pred_270 = np.rot90(pred_270, 1)
        pred_f = np.fliplr(pred_f)
        pred_90f = np.rot90(np.fliplr(pred_90f), 3)
        pred_180f = np.rot90(np.fliplr(pred_180f), 2)
        pred_270f = np.rot90(np.fliplr(pred_270f), 1)
        prediction = (pred + pred_90 + pred_180 + pred_270 + pred_f + pred_90f + pred_180f + pred_270f) * 255.0 / 8.0
        prediction = prediction.clip(0, 255)

        Image.fromarray(np.uint8(prediction)).save(SR_image[i])

