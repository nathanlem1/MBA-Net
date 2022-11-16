import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

from MBA_modules import SAM_module, CAM_module
# from MBA.MBA_modules import SAM_module, CAM_module


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    # print(class_name)
    if class_name.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        print(class_name)
    elif class_name.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif class_name.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif class_name.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Define a new FC layer and classification layer - |--Linear--|--bn--|--relu--|--dropout--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, drop_rate, relu=True, batch_norm=True, num_bottleneck=2048, linear=True):  # 512
        super(ClassBlock, self).__init__()
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if batch_norm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
            # add_block += [nn.ReLU()]
        if drop_rate > 0:
            add_block += [nn.Dropout(p=drop_rate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


# Define an Identity layer to replace the FC layer of the pretrained model
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Define the ResNet50-based MBA (Multi-Branch with Attention) model
class ResNet50_MBA(nn.Module):

    def __init__(self, class_num, drop_rate=0.5, stride=1, relative_pos=True, part_h=3, part_v=1, use_attention=True):
        super(ResNet50_MBA, self).__init__()
        backbone = models.resnet50(pretrained=True)
        backbone.fc = Identity()
        self.part_h = part_h
        self.part_v = part_v
        self.use_attention = use_attention

        # Remove the final down sample which increases performance by changing the last stride from 2 to 1.
        if stride == 1:
            backbone.layer4[0].conv2.stride = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = backbone
        self.avgpool_p = nn.AdaptiveAvgPool2d((self.part_h, self.part_v))  # horizontal and vertical parts. For GAP,
        # part_h = part_v = 1, for conventional average pooling (AG), part_h > 1 or part_v > 1 (part_h*part_v > 1).

        # For SAM
        self.layer2s = backbone.layer2
        self.layer3s = backbone.layer3
        self.layer4s = backbone.layer4
        self.avgpools = backbone.avgpool

        self.backbone.layer2s = backbone.layer2
        self.backbone.layer3s = backbone.layer3
        self.backbone.layer4s = backbone.layer4
        self.backbone.avgpools = backbone.avgpool

        # For CAM
        self.layer2c = backbone.layer2
        self.layer3c = backbone.layer3
        self.layer4c = backbone.layer4
        self.avgpoolc = backbone.avgpool

        self.backbone.layer2c = backbone.layer2
        self.backbone.layer3c = backbone.layer3
        self.backbone.layer4c = backbone.layer4
        self.backbone.avgpoolc = backbone.avgpool

        # SAM for layer 3 and layer 4 of ResNet50
        # self.sam_att1 = SAM_module(256, 81, relative_pos=relative_pos)  # SAM, for 324x324 input
        # self.sam_att2 = SAM_module(512, 41, relative_pos=relative_pos)
        self.sam_att3 = SAM_module(1024, 21, relative_pos=relative_pos)
        self.sam_att4 = SAM_module(2048, 21, relative_pos=relative_pos)

        # CAM for layer 3 and layer 4 of ResNet50
        # self.cam_att1 = CAM_module(81*81)  # CAM
        # self.cam_att2 = CAM_module(41*41)
        self.cam_att3 = CAM_module(21*21)
        self.cam_att4 = CAM_module(21*21)

        # # Initialize parameters - using this gives less result!
        # self.sam_att1.apply(weights_init_kaiming)
        # self.sam_att2.apply(weights_init_kaiming)
        # self.sam_att3.apply(weights_init_kaiming)
        # self.sam_att4.apply(weights_init_kaiming)

        # Reduction layers
        # Define part_h*part_v classifiers
        if self.part_h * self.part_v > 1:
            for i in range(self.part_h * self.part_v):
                name = 'reduction_p' + str(i)
                setattr(self, name, ClassBlock(2048, class_num, drop_rate))
        # Define Global, spatial attention and channel attention classifiers
        self.reduction_g = ClassBlock(2048, class_num, drop_rate)
        if self.use_attention:
            self.reduction_s = ClassBlock(2048, class_num, drop_rate)
            self.reduction_c = ClassBlock(2048, class_num, drop_rate)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        # Copy for each branch
        xs = xc = x

        # Get feature maps after layer 4
        xg = xp = self.backbone.layer4(x)

        # Part or Local branch (without attention)
        num_classifiers = self.part_h * self.part_v  # Total number classifiers (total number of partitions).
        xp_all = []  # list of predictions
        if num_classifiers > 1:
            xp = self.avgpool_p(xp)  # AG
            feat_xp = xp.view(xp.size(0), xp.size(1), xp.size(2) * xp.size(3))  # Features after conventional AG
            for i in range(num_classifiers):
                part_i = torch.squeeze(feat_xp[:, :, i])
                name = 'reduction_p' + str(i)
                cls = getattr(self, name)
                predict = cls(part_i)  # Predictions
                xp_all.append(predict)
        else:
            # print('Local (part) classifiers are not used. To use them, num_classifiers (part_h * part_v) MUST be '
            #       'greater than 1!')
            feat_xp = []

        # Global (without attention) branch
        xg = self.backbone.avgpool(xg)  # GAP
        feat_xg = xg.view(xg.size(0), xg.size(1))  # Features just after GAP (Global Average Pooling)
        # feat_bef_x = self.reduction_g.add_block[0](feat_x)  # Features before batchnorm
        # feat_aft_x = self.reduction_g.add_block[:2](feat_x)  # Features after batchnorm
        xg = self.reduction_g(feat_xg)  # Predictions

        if self.use_attention:
            # Spatial attention branch
            xs = self.sam_att3(xs)
            xs = self.backbone.layer4s(xs)
            xs = self.sam_att4(xs)
            xs = self.backbone.avgpools(xs)  # GAP
            feat_xs = xs.view(xs.size(0), xs.size(1))   # Features just after GAP
            # feat_bef_xs = self.reduction_s.add_block[0](feat_xs)  # Features before batchnorm
            # feat_aft_xs = self.reduction_s.add_block[:2](feat_xs)  # Features after batchnorm
            xs = self.reduction_s(feat_xs)   # Predictions

            # Channel attention branch
            xc = self.cam_att3(xc)
            xc = self.backbone.layer4c(xc)
            xc = self.cam_att4(xc)
            xc = self.backbone.avgpoolc(xc)  # GAP
            feat_xc = xc.view(xc.size(0), xc.size(1))   # Features just after GAP
            # feat_bef_xc = self.reduction_c.add_block[0](feat_xc)  # Features before batchnorm
            # feat_aft_xc = self.reduction_c.add_block[:2](feat_xc)  # Features after batchnorm
            xc = self.reduction_c(feat_xc)   # Predictions

            out = {
                'x': (xp_all, xg, xs, xc),
                'features': (feat_xp, feat_xg, feat_xs, feat_xc),  # Local (parts), Global & attentions
            }
        else:
            out = {
                'x': (xp_all, xg),
                'features': (feat_xp, feat_xg),   # Only Local (parts) & Global
            }

        return out


# Check the model
if __name__ == '__main__':
    input_ = torch.FloatTensor(4, 3, 324, 324)

    # ResNet50-based MBA model
    model = ResNet50_MBA(200)
    print(model)
    outputs = model(input_)
    print('Output size:', outputs.keys())

    print('ok')
