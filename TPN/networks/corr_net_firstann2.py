import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.modules as modules
from correlation_package.modules.correlation import Correlation

class corr_net(nn.Module):
    def __init__(self, init_weight=True):
        super(corr_net, self).__init__()
        self.conv_in_a = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.conv_in_b = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.conv_in_c = nn.Conv2d(512, 512, kernel_size=1, bias=True)

        # corr feat
        self.corr12 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr13 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr23 = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.corr_conv_out = nn.Conv2d(441 * 3, 3, kernel_size=3, padding=1, bias=True)
        self.corr_feat = nn.Sequential(
            nn.Linear(600, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )

        # concat feat
        self.concat_conv_out = nn.Conv2d(1536, 3, kernel_size=3, padding=1, bias=True)
        self.concat_feat = nn.Sequential(
            nn.Linear(600, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )

        self.linear_out = nn.Linear(256, 2)
        if init_weight:
            self._initialize_weights()

    def forward(self, feat1, feat2, feat3):
        feat1 = self.conv_in_a(feat1)
        feat2 = self.conv_in_b(feat2)
        feat3 = self.conv_in_c(feat3)

        # correlation branch
        out_corr12 = self.corr12(feat1, feat2)
        out_corr13 = self.corr13(feat1, feat3)
        out_corr23 = self.corr23(feat2, feat3)
        out_corr = torch.cat((out_corr12, out_corr13, out_corr23), 1)
        out_corr = self.corr_activation(out_corr)
        out_corr = self.corr_conv_out(out_corr)
        out_corr = nn.ReLU(True)(out_corr)

        out_corr = out_corr.view(out_corr.size(0), -1)
        out_corr = self.corr_feat(out_corr)

        # concat branch
        out_cat = torch.cat((feat1, feat2, feat3), 1)
        out_cat = self.concat_conv_out(out_cat)
        out_cat = out_cat.view(out_cat.size(0), -1)
        out_cat = self.concat_feat(out_cat)

        # fusion
        output = torch.cat((out_corr, out_cat), 1)
        output = self.linear_out(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

