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
    def __init__(self, roi_size=(12, 12)):
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

        self.roi_pool = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
                        nn.Linear(256*self.roi_size[0]*self.roi_size[1], 4096), 
                        nn.ReLU(inplace=True), 
                        nn.Linear(4096, 1024), 
                        nn.ReLU(inplace=True))

        self.score_fc   = nn.Sequential(
                            nn.Linear(1024, self.n_classes),
                            nn.Softmax(dim=1))

        self.bbox_fc    = nn.Sequential(
                            nn.Linear(1024, self.n_classes),
                            nn.Softmax(dim=0))

    def forward(self, x, rois=None):
        out = self.backbone(x)
        conv_features = self.encoder(out)
        h, w = conv_features.shape[2], conv_features.shape[3]
        spp_output = self.roi_pool(conv_features, [rois], self.roi_size, (h/x.shape[2], w/x.shape[3]))
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        class_scores = self.score_fc(classifier_ouput)
        class_scores = nn.functional.softmax(class_scores, dim=1)
        
        bbox_scores = self.bbox_fc(classifier_ouput)
        bbox_scores = nn.functional.softmax(bbox_scores, dim=0)

        cls_prob = class_scores * bbox_scores
        return cls_prob 

if __name__ == '__main__':
    x = torch.randn((1, 36, 800, 700)).cuda()
    model = WSDDNPIXOR()
    model = model.cuda()
    out, conv_features = model(x)
    print(out.shape, conv_features.shape)