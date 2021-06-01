import torch
import torch.nn as nn
import torchvision.models as models

from corr_net_firstann2_davis16 import corr_net

class vgg_flowc(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_flowc, self).__init__()
        self.feat_extractor = models.vgg16(pretrained=pretrained).features
        self.corr_net = corr_net()

    def forward(self, x):
        first_img = x[:, 0:3, :, :]
        first_ann = x[:, 3:4, :, :]
        image1 = x[:, 4:7, :, :]
        image2 = x[:, 7:10, :, :]

        feat1 = self.feat_extractor.forward(image1)
        feat2 = self.feat_extractor.forward(image2)
        feat3 = self.feat_extractor.forward(first_img * first_ann)

        pred = self.corr_net.forward(feat1, feat2, feat3)

        return pred


