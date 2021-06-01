import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.modules as modules
from osvos_layers import center_crop, interp_surgery

class OSVOS(nn.Module):
    def __init__(self, pretrained=True, in_channel=4):
        super(OSVOS, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        lay_list = [[2, 4],
                    [4, 9],
                    [9, 16],
                    [16, 23],
                    [23, 30]]
        side_channels = [64, 128, 256, 512, 512]
        print("constructing OSVOS architecture..")
        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()

        # construct the network
        for i in range(len(lay_list)):
            stages.append(nn.Sequential(*[model.features[j] for j in range(lay_list[i][0], lay_list[i][1])]))

            if i > 0:
                side_prep.append(nn.Conv2d(side_channels[i], 16, kernel_size=3, padding=1))
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn

        self.fuse = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0))

        print("Initializing weights..")
        self._initialize_weights([self.conv1, self.upscale, self.upscale_, self.side_prep, self.score_dsn, self.fuse])

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.conv1(x)
        x = self.stages[0](x)

        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))

        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def _initialize_weights(self, mods):
        for s in mods:
            for m in s:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.001)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.zero_()
                    m.weight.data = interp_surgery(m)
