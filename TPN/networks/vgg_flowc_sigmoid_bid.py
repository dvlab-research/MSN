import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.modules as modules
from vgg_osvos import OSVOS
from flownet_models import FlowNet2C
from proposal_select2 import proposal_select

from channelnorm_package.modules.channelnorm import ChannelNorm

class vgg_flowc(nn.Module):
    def __init__(self, args=None, pretrained=True, pre_norm=True):
        super(vgg_flowc, self).__init__()
        self.vgg_rgb = OSVOS(pretrained, 4)
        self.vgg_flow = OSVOS(pretrained, 2)
        self.proposal_branch = proposal_select(in_channel=2)
        self.flownetc = FlowNet2C(args, pre_norm=pre_norm)
        self.side_fusion = modules.ModuleList()

        for i in range(6):
            self.side_fusion.append(nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)
            ))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        image1 = x[:, 0:3, :, :]
        image2 = x[:, 3:6, :, :]
        label1 = x[:, 6:7, :, :]

        # get flow
        image21 = torch.cat((image2.unsqueeze(2), image1.unsqueeze(2)), 2)
        flow21 = ChannelNorm()(self.flownetc.forward(image21)[0])

        # for two branch
        pred_rgb = self.vgg_rgb.forward(torch.cat((image2, label1), 1))
        pred_flow = self.vgg_flow.forward(torch.cat((flow21, label1), 1))
        pred_proposal = self.proposal_branch.forward(label1, flow21)
        pred = []

        for i in range(len(pred_rgb)):
            temp = self.side_fusion[i].forward(torch.cat((pred_rgb[i], pred_flow[i], pred_proposal), 1))
            pred.append(self.sigmoid(temp))

        return pred


