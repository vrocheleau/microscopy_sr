import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):

        # If grayscale copy in RGB channels
        if high_resolution.shape[1] == 1:
            high_resolution = torch.cat((high_resolution, high_resolution, high_resolution), dim=1)
            fake_high_resolution = torch.cat((fake_high_resolution, fake_high_resolution, fake_high_resolution), dim=1)

        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss

