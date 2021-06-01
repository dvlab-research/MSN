import torch
import torch.nn as nn
#import torchvision.models as models

#import torch.nn.modules as modules

class proposal_select(nn.Module):
    def __init__(self, in_channel=2, conv_num = 3):
        super(proposal_select, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.block_list = []
        for i in range(conv_num):
            conv2d = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
            self.block_list += [conv2d, nn.ReLU(inplace=True)]

        self.blocks = nn.Sequential(*self.block_list)

        self.conv_out = nn.Conv2d(32, 1, kernel_size=1, bias=False)

    def forward(self, label, flow):
        x = torch.cat((label, flow), 1)
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv_out(x)

        return x

