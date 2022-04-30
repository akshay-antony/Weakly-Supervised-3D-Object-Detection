import imp
from turtle import forward
import torch, torchvision
import torch.nn as nn


class WSDNN_Resnet(nn.Module):
    def __init__(self, roi_size=(18, 6)):
        super(WSDNN_Resnet, self).__init__()
        self.roi_size = roi_size
        self.num_class = 1
        resnet = torchvision.models.resnet152(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self.roi_pool = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
                        nn.Linear(2048*self.roi_size[0]*self.roi_size[1], 4096), 
                        nn.ReLU(inplace=True), 
                        nn.Linear(4096, 4096), 
                        nn.ReLU(inplace=True))

        self.score_fc   = nn.Sequential(
                            nn.Linear(4096, self.n_classes),
                            nn.Softmax(dim=1))

        self.bbox_fc    = nn.Sequential(
                            nn.Linear(4096, self.n_classes),
                            nn.Softmax(dim=0))
        
    def forward(self, x, rois):
        out = self.encoder(x)
        h, w = out.shape[2], out.shape[3]
        spp_output = self.roi_pool(out, [rois], self.roi_size, h/x.shape[2])
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        class_scores = self.score_fc(classifier_ouput)
        bbox_scores = self.bbox_fc(classifier_ouput)
        cls_prob = class_scores * bbox_scores 
        return cls_prob.reshape(cls_prob.shape[0], -1, self.num_class)

if __name__ == '__main__':
    model = WSDNN_Resnet()
    x = torch.randn((1, 3, 1224, 370))
    with torch.no_grad():
        out = model(x)
        print(out.shape)