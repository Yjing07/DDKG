import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch import nn
from torch.nn import functional as F


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.boundary = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature): 
        low_level_feature = self.project(feature['low_level']) 
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False) 
        mask_channel304 = torch.cat([low_level_feature, output_feature], dim=1) 
        mask_channel1 = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
        return {'mask_channel304': mask_channel304, 
                'mask_channel1': mask_channel1,
                'boundary': self.boundary(output_feature)}

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):# feature['out']是[512 16 16]
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return

class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(CrossAttentionModule, self).__init__()
        self.in_channels = in_channels
        

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(1, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature_map, attention_map):
        b, c, h, w = feature_map.size()
        
        query = self.query_conv(feature_map).view(b, c, -1)
        key = self.key_conv(attention_map).view(b, c, -1).permute(0, 2, 1) 
        value = self.value_conv(feature_map).view(b, c, -1) 
        
        attention_scores = torch.bmm(query, key) 
        attention_probs = nn.functional.softmax(attention_scores, dim=2)

        out = torch.bmm(attention_probs, value) 
        out = out.view(b, c, h, w)
        
        out = self.gamma * out + feature_map
        
        return out
    
class SegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.attention = CrossAttentionModule(in_channels=2048)
        self.downsample_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, stride=8, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(304, 2)


    def forward(self, x):
        input_shape = x.shape[-2:]
        """print("hehe")
        print(input_shape)
        print("wawawa")"""
        features = self.backbone(x)
        x = self.classifier(features)

        _x = x['mask_channel304'] + torch.sigmoid(x['mask_channel1']) * x['mask_channel304'] 
        mask = F.interpolate(x['mask_channel1'], size=input_shape, mode='bilinear', align_corners=False)    
        x = self.avgpool(_x)
        x_304 = torch.flatten(x, 1)
        output = self.fc(x_304)

        
        return mask, output



if __name__ == '__main__':
    from models.resnet import resnet34 

    #bkbone = resnet34().cuda()
    #head = DeepLabHeadV3Plus(in_channels=512, low_level_channels=64, num_classes=1)
    #model = SegmentationModel(bkbone, head).cuda()

    # from backbone.xception import xception #networks.
    # bkbone = xception(pretrained=False).cuda()
    # head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=64,num_classes=1)
    # model =  SegmentationModel(bkbone,head).cuda()

    # from backbone.mobilenetv2 import mobilenet_v2 #networks.
    # bkbone = mobilenet_v2(pretrained=False).cuda()
    # head = DeepLabHeadV3Plus(in_channels=1280,low_level_channels=24,num_classes=1)
    # model =  SegmentationModel(bkbone,head).cuda()

    # from backbone.vgg import VGG16 #networks.
    # bkbone = VGG16(pretrained=False).cuda()
    # head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=1)
    # model =  SegmentationModel(bkbone,head).cuda()

    #from backbone.pvt import pvt_v2_b2 #networks.
    #bkbone = pvt_v2_b2().cuda()
    #head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=1)
    #model =  SegmentationModel(bkbone,head).cuda()

    """img = torch.randn((2, 3, 384, 384)).cuda()
    mask_pre = model(img)
    
    print(img.shape)
    print(mask_pre)
    print(mask_pre.shape)"""
    
    

    