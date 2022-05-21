from requests import patch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch
from torch.nn import functional as F
import timm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelWeight(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ChannelWeight, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        return out

class MSALayer(nn.Module):

    def __init__(self, channel, reduction=8):
        super(MSALayer, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(2 * channel, channel, 1)
        self.bn = nn.BatchNorm2d(channel)
        self.gelu = nn.GELU()
        self.se = ChannelWeight(channel, reduction)
       # self.out_channel = channel // 2
        self.softmax = nn.Softmax(dim=1)
        self.channel = channel

    def forward(self, x):
        b,c,h,w = x.size()
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        y = torch.cat((x1, x2), dim=1)
        y = y.view(b, 2, self.channel, h, w)

        se_x1 = self.se(x1)
        se_x2 = self.se(x2)

        se_x = torch.cat((se_x1, se_x2), dim=1)
        atten = se_x.view(b, 2, self.channel, 1, 1)
        atten = self.softmax(atten)
        z = y * atten
        for i in range(2):
            group_y = z[:, i, :, :]
            if i == 0:
                out = group_y
            else:
                out = torch.cat((group_y, out), 1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.gelu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)






class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class SRM(nn.Module):
    def __init__(self, channel, reduction=None):
        super(SRM, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)

def Conv1dBlock(inplane, outplane, kernel_size, padding):
    res = nn.Sequential(
        nn.Conv1d(inplane, outplane, kernel_size, padding=padding),
        nn.BatchNorm1d(outplane),
        nn.GELU()
    )
    return res
    



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16, patch_size=2, num_patches=256):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256, reduction=16):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out = self.sa(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16, patch_size=2, num_patches=256):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out)
        out = self.sa(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256,reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class MSABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256,reduction=8):
        super(MSABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.msa = MSALayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.msa(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class CABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16, patch_size=2, num_patches=256):
        super(CABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class SRMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256,reduction=16):
        super(SRMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.srm = SRM(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.srm(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class SRMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16, patch_size=2, num_patches=256):
        super(SRMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.srm = SRM(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.srm(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16, patch_size=2, num_patches=256):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class GAMLayer(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAMLayer, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class GABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256, reduction=16):
        super(GABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ga = GAMLayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ga(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class PABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, patch_size=2, num_patches=256, reduction=16):
        super(PABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pa = PALayer(planes, patch_size, num_patches, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pa(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out



class PABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=8, patch_size=3, num_patches=256):
        super(PABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.pa = PALayer(planes * 4, patch_size, num_patches)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.pa(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], num_patches=256)
        self.layer2 = self._make_layer(block, 128, layers[1], num_patches=64, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_patches=16, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_patches=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, num_patches=256,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                         num_patches=num_patches))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_patches=num_patches))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x

def srmresnet18(num_classes=10):
    
    model = ResNet(SRMBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def srmresnet34(num_classes=10):
    
    model = ResNet(SRMBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def srmresnet50(num_classes=10):
    
    model = ResNet(SRMBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def msaresnet18(num_classes=10):
    
    model = ResNet(MSABasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def msaresnet34(num_classes=10):
    
    model = ResNet(MSABasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def garesnet18(num_classes=10):
    
    model = ResNet(GABasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet18(num_classes=10):
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet34(num_classes=10):
    
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50(num_classes=10):
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def cbam_resnet18(num_classes=10):
    
    model = ResNet(CBAMBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def cbam_resnet34(num_classes=10):
   
    model = ResNet(CBAMBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def cbam_resnet50(num_classes=10):
   
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def se_resnet18(num_classes=10):
   
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def se_resnet34(num_classes=10):
    
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def se_resnet50(num_classes=10):
    
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def pa_resnet18(num_classes=10):
   
    model = ResNet(PABasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def pa_resnet34(num_classes=10):
  
    model = ResNet(PABasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def pa_resnet50(num_classes=2):
   
    model = ResNet(PABottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


class PALayer(nn.Module):
    def __init__(self, channel, patch_size=1, num_patches=9, expansion=8, reduction=16):
        super(PALayer, self).__init__()
        
        self.proj = nn.Conv2d(channel, channel // reduction, kernel_size=patch_size, stride=patch_size)
        self.conv1 = nn.Conv1d(channel // reduction, channel, kernel_size=3, stride=1, padding=1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_patches, num_patches * expansion, bias=False),
            nn.GELU(),
            nn.LayerNorm(num_patches * expansion),
            nn.Linear(num_patches * expansion, num_patches, bias=False),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.proj(x).flatten(2)
        y = self.conv1(y)
        y1 = torch.mean(y, dim=1, keepdim=True)
        _, y2 = torch.max(y, dim=1, keepdim=True)
        y = y1 + y2
        y = self.fc(y)
        y = y.reshape(b, 1, h, w)
        out = y * x
        return out

class ResNept(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNept, self).__init__()
        resnet = timm.create_model("resnet34", pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.act1 
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.global_pool

        self.res1 = resnet.layer1 
        self.res2 = resnet.layer2 
        self.res3 = resnet.layer3 
        self.res4 = resnet.layer4 
        self.fc = nn.Linear(512, num_classes)
        self.pa = PALayer(512)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pa(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class PADensenet(nn.Module):
    def __init__(self, num_classes=2):
        super(PADensenet, self).__init__()
        densenet = timm.create_model("densenet121", pretrained=True)
        self.features = densenet.features
        self.avgpool = densenet.global_pool
        self.pa = PALayer(1024)
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pa(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x   



class PARes2net(nn.Module):
    def __init__(self, num_classes=2):
        super(PARes2net, self).__init__()
        resnet = timm.create_model("res2net50_14w_8s", pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.act1 
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.global_pool

        self.res1 = resnet.layer1 
        self.res2 = resnet.layer2 
        self.res3 = resnet.layer3 
        self.res4 = resnet.layer4 
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.pa = PALayer(2048)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pa(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x