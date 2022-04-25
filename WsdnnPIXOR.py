from turtle import forward
import torch
import numpy as np
import torch.nn as nn
from model import BackBone, Bottleneck
import torchvision

geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'input_shape': (800, 700, 36),
        'label_shape': (200, 175, 7)
    }

class WSDDNPIXOR(nn.Module):
    def __init__(self, 
                roi_size=(12, 12)):
        super(WSDDNPIXOR, self).__init__()
        self.roi_size = roi_size
        self.n_classes = 9
        self.backbone = BackBone(Bottleneck, 
                                [3, 6, 6, 3], 
                                geometry, 
                                use_bn=True)

        self.encoder = nn.Sequential(
                            nn.Conv2d(96, 256, (3, 3), (1, 1), (1, 1)),
                            nn.ReLU(),
                            nn.MaxPool2d((3,3), (2, 2), (1, 1)),
                            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                            nn.ReLU(),
                            nn.MaxPool2d((3, 3), (2, 2), (1, 1)))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((40, 35))
        self.roi_pool = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
                        nn.Linear(256*self.roi_size[0]*self.roi_size[1], 2*4096), 
                        nn.ReLU(inplace=True), 
                        nn.Linear(2*4096, 2*1024), 
                        nn.ReLU(inplace=True))

        self.score_fc   = nn.Sequential(
                            nn.Linear(2*1024, self.n_classes),
                            nn.Softmax(dim=1))

        self.bbox_fc    = nn.Sequential(
                            nn.Linear(2*1024, self.n_classes),
                            nn.Softmax(dim=0))

    def forward(self, 
                x, 
                rois=None,
                should_return=False):
        out = self.backbone(x)
        conv_features = self.encoder(out)
        conv_features = self.adaptive_pool(conv_features)
        h, w = conv_features.shape[2], conv_features.shape[3]
        spp_output = self.roi_pool(conv_features, [rois], self.roi_size, (h/x.shape[2]))
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        class_scores = self.score_fc(classifier_ouput)
        bbox_scores = self.bbox_fc(classifier_ouput)
        cls_prob = class_scores * bbox_scores 
        return cls_prob 

if __name__ == '__main__':
    pass
    # x = torch.randn((1, 36, 800, 700)).cuda()
    # model = WSDDNPIXOR()
    # model = model.cuda()
    # rois = torch.randn((100, 4)).cuda()
    # out = model(x, rois)
    # print(out.shape)