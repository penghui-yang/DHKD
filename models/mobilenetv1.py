import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import NonLinearClassifier


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, dual_head=False, aux_head_linear=True):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        
        self.classifier = nn.Linear(1024, num_classes)
        self.dual_head = dual_head
        if self.dual_head:
            Classifier_2 = nn.Linear if aux_head_linear else NonLinearClassifier
            self.classifier2 = Classifier_2(1024, num_classes)

    def forward(self, x, is_feat=False):
        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](F.relu(feat3)))
        feat5 = self.model[14](F.relu(feat4))
        avg = feat5.reshape(-1, 1024)
        out1 = self.classifier(avg)
        if self.dual_head:
            out2 = self.classifier2(avg)
            out = [out1, out2]
        else:
            out = out1

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [F.relu(feat1), F.relu(feat2), F.relu(feat3), F.relu(feat4)]
        feats["preact_feats"] = [feat1, feat2, feat3, feat4]
        return (feats, out) if is_feat else out

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 1024]

def mobilenetv1(num_classes, dual_head=False, aux_head_linear=True):
    return MobileNetV1(num_classes=num_classes, dual_head=dual_head, aux_head_linear=aux_head_linear)
