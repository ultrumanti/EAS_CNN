import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torch.nn.init
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import BatchNorm2d
from functools import partial
from torchvision import models


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

nonlinearity = partial(F.relu, inplace=True)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan,
                      out_chan,
                      kernel_size=ks,
                      stride=stride,
                      padding=padding,
                      bias=False),
            BatchNorm2d(out_chan),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.l2_norm = l2_norm
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc11 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc12 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        self.fc21 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc22 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        self.relu1 = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes)
    )

class CNN_Block(nn.Module):
    def __init__(self, in_planes, out_planes, flag_cov):
        super(CNN_Block, self).__init__()
        self.flag_cov = flag_cov
        self.in_planes = in_planes
        self.out_planes = out_planes

        if self.flag_cov == "AC_Block_3":
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
            self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
            self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
            self.bn = nn.BatchNorm2d(out_planes)
            self.ReLU = nn.ReLU(True)

        elif self.flag_cov == "CNN_Block_3":
            self.conv1 = conv3x3(in_planes, out_planes)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=True)

        elif self.flag_cov == "AC_Block_5":
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=5, padding=2, stride=1)
            self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 5), padding=(0, 2), stride=1)
            self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(5, 1), padding=(2, 0), stride=1)
            self.bn = nn.BatchNorm2d(out_planes)
            self.ReLU = nn.ReLU(True)

        elif self.flag_cov == "CNN_Block_5":
            self.conv1 = conv3x3(in_planes, out_planes, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.flag_cov == "AC_Block_3":
            x1 = self.squre(x)
            x2 = self.cross_ver(x)
            x3 = self.cross_hor(x)
            x4 = self.ReLU(self.bn(x1 + x2 + x3))

        elif self.flag_cov == "CNN_Block_3":
            out = self.conv1(x)
            out = self.bn1(out)
            x4 = self.relu(out)

        elif self.flag_cov == "AC_Block_5":
            x1 = self.squre(x)
            x2 = self.cross_ver(x)
            x3 = self.cross_hor(x)
            x4 = self.ReLU(self.bn(x1 + x2 + x3))

        elif self.flag_cov == "CNN_Block_5":
            out = self.conv1(x)
            out = self.bn1(out)
            x4 = self.relu(out)
        return x4


class AttentionEnhancementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionEnhancementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = PAM_Module(out_chan)
        self.bn_atten = BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        att = self.conv_atten(feat)
        return self.bn_atten(att)


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = PAM_Module(out_chan)

    def forward(self, fcat):
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out


class EV_Unet(nn.Module):
    def __init__(self, band_num, class_num, model_settings):
        super(EV_Unet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'EV_Unet'
        self.model_settings = model_settings
        channels_end = 16

        channels = [64, 128, 256, 512, 1024]
        resnet = models.resnet18(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.conv1 = resnet.layer1

        # <editor-fold desc=" encoder_1 ">
        self.conv12 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[0], channels[1]),
            CNN_Block(channels[1], channels[1], self.model_settings['Encoder_1'])
        )
        self.conv13 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[1], channels[2]),
            CNN_Block(channels[2], channels[2], self.model_settings['Encoder_1']),
        )
        self.conv14 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[2], channels[3]),
            CNN_Block(channels[3], channels[3], self.model_settings['Encoder_1'])
        )
        # </editor-fold>

        # <editor-fold desc=" encoder_2 ">
        self.conv2 = resnet.layer2

        self.conv23 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[1], channels[2]),
            CNN_Block(channels[2], channels[2], self.model_settings['Encoder_2'])
        )
        self.conv24 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[2], channels[3]),
            CNN_Block(channels[3], channels[3], self.model_settings['Encoder_2'])
        )
        # </editor-fold>

        # <editor-fold desc=" encoder_3 ">
        self.conv3 = resnet.layer3

        self.conv34 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[2], channels[3]),
            CNN_Block(channels[3], channels[3], self.model_settings['Encoder_3'])
        )
        # </editor-fold>

        # <editor-fold desc=" encoder_4 ">
        self.conv4 = resnet.layer4
        # </editor-fold>

        # <editor-fold desc=" encoder_5 ">
        self.conv5_redu = resnet.layer4

        self.conv5_1_redu = nn.Sequential(
            CNN_Block(channels[3], channels[3], self.model_settings['Encoder_5'])
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv1x1(channels[3], channels[4]),
            CNN_Block(channels[4], channels[4], self.model_settings['Encoder_5']),
        )
        self.conv5_1 = nn.Sequential(
            CNN_Block(channels[4], channels[4], self.model_settings['Encoder_5'])
        )
        # </editor-fold>

        # <editor-fold desc=" encoder_5 ">
        self.attention5_to_5_redu = PAM_CAM_Layer(channels[3])
        self.attention5_to_5 = PAM_CAM_Layer(channels[4])
        # </editor-fold>

        # <editor-fold desc=" decoder_4 ">
        self.attention4_0 = PAM_CAM_Layer(channels[3])

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv43 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv42 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv41 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.attention4_1 = PAM_CAM_Layer(channels[3])
        self.attention4_1_b = PAM_CAM_Layer(channels[4])
        self.skblock4_1 = ChannelAttention(channels[3], channels[3], 16)
        self.skblock4_1_b = ChannelAttention(channels[4], channels[4], 16)

        self.conv_deco_4_1 = nn.Sequential(
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
        )
        self.conv_deco_4_2 = nn.Sequential(
            conv1x1(channels[4], channels[3]),
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
        )
        self.to_decoder_4_more_two_tensor = self.model_settings['to_decoder_4_more_two_tensor']
        self.to_decoder_4_more_two_tensor_arry = self.to_decoder_4_more_two_tensor.split("]--[")

        if self.to_decoder_4_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "full":
            self.count_list_4 = self.to_decoder_4_more_two_tensor_arry.count('con') + 1
        elif self.to_decoder_4_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "reduce_one":
            self.count_list_4 = self.to_decoder_4_more_two_tensor_arry.count('con')
        else:
            self.count_list_4 = self.to_decoder_4_more_two_tensor_arry.count('con') + 1

        self.attention4_2 = PAM_CAM_Layer(channels[3] * self.count_list_4)
        self.skblock4_2 = ChannelAttention(channels[3] * self.count_list_4, channels[3] * 2, 16)
        self.AABlock_deco_4_4_add = AttentionAggregationModule(channels[3] * (self.count_list_4), channels[3] * 2)

        self.conv_deco_4_3 = nn.Sequential(
            CNN_Block(channels[3] * self.count_list_4, channels[3], self.model_settings['to_decoder_4_conv']),
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
        )
        self.conv_deco_4_4 = nn.Sequential(
            CNN_Block(channels[3] * 2, channels[3], self.model_settings['to_decoder_4_conv']),
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
        )
        self.conv_deco_4_4_no_att = nn.Sequential(
            CNN_Block(channels[3] * self.count_list_4, channels[3] * 2, self.model_settings['to_decoder_4_conv']),
            CNN_Block(channels[3] * 2, channels[3], self.model_settings['to_decoder_4_conv']),
            CNN_Block(channels[3], channels[3], self.model_settings['to_decoder_4_conv']),
        )
        # </editor-fold>

        # <editor-fold desc=" decoder_3 ">
        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv32 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv31 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.attention3_0 = PAM_CAM_Layer(channels[2])
        self.attention3_1 = PAM_CAM_Layer(channels[2])
        self.skblock3_1 = ChannelAttention(channels[2], channels[2], 16)
        self.attention3_1_b = PAM_CAM_Layer(channels[3])
        self.skblock3_1_b = ChannelAttention(channels[3], channels[3], 16)

        self.conv_deco_3_1 = nn.Sequential(
            CNN_Block(channels[2], channels[2], self.model_settings['to_decoder_3_conv']),
        )
        self.conv_deco_3_2 = nn.Sequential(
            CNN_Block(channels[3], channels[2], self.model_settings['to_decoder_3_conv']),
            CNN_Block(channels[2], channels[2], self.model_settings['to_decoder_3_conv']),
        )
        self.to_decoder_3_more_two_tensor = self.model_settings['to_decoder_3_more_two_tensor']
        self.to_decoder_3_more_two_tensor_arry = self.to_decoder_3_more_two_tensor.split("]--[")

        if self.to_decoder_3_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "full":
            self.count_list_3 = self.to_decoder_3_more_two_tensor_arry.count('con') + 1
        elif self.to_decoder_3_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "reduce_one":
            self.count_list_3 = self.to_decoder_3_more_two_tensor_arry.count('con')
        else:
            self.count_list_3 = self.to_decoder_3_more_two_tensor_arry.count('con') + 1
        self.attention3_2 = PAM_CAM_Layer(channels[2] * self.count_list_3)
        self.skblock3_2 = ChannelAttention(channels[2] * self.count_list_3, channels[2] * 2, 16)

        self.conv_deco_3_3 = nn.Sequential(
            CNN_Block(channels[2] * self.count_list_3, channels[2], self.model_settings['to_decoder_3_conv']),
            CNN_Block(channels[2], channels[2], self.model_settings['to_decoder_3_conv']),
        )
        self.conv_deco_3_4 = nn.Sequential(
            CNN_Block(channels[2] * 2, channels[2], self.model_settings['to_decoder_3_conv']),
            CNN_Block(channels[2], channels[2], self.model_settings['to_decoder_3_conv']),
        )
        self.AABlock_deco_3_4 = AttentionAggregationModule(channels[2] * (self.count_list_3), channels[2] * 2)
        # </editor-fold>

        # <editor-fold desc=" decoder_2 ">
        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv21 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.attention2_0 = PAM_CAM_Layer(channels[1])
        self.attention2_1 = PAM_CAM_Layer(channels[1])
        self.attention2_1_b = PAM_CAM_Layer(channels[2])
        self.skblock2_1 = ChannelAttention(channels[1], channels[1], 16)
        self.skblock2_1_b = ChannelAttention(channels[2], channels[2], 16)

        self.conv_deco_2_1 = nn.Sequential(
            CNN_Block(channels[1], channels[1], self.model_settings['to_decoder_2_conv']),
        )
        self.conv_deco_2_2 = nn.Sequential(
            CNN_Block(channels[2], channels[1], self.model_settings['to_decoder_2_conv']),
            CNN_Block(channels[1], channels[1], self.model_settings['to_decoder_2_conv']),
        )
        self.to_decoder_2_more_two_tensor = self.model_settings['to_decoder_2_more_two_tensor']
        self.to_decoder_2_more_two_tensor_arry = self.to_decoder_2_more_two_tensor.split("]--[")

        if self.to_decoder_2_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "full":
            self.count_list_2 = self.to_decoder_2_more_two_tensor_arry.count('con') + 1
        elif self.to_decoder_2_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "reduce_one":
            self.count_list_2 = self.to_decoder_2_more_two_tensor_arry.count('con')
        else:
            self.count_list_2 = self.to_decoder_2_more_two_tensor_arry.count('con') + 1

        self.AABlock_deco_2_4 = AttentionAggregationModule(channels[1] * (self.count_list_2), channels[1] * 2)
        self.attention2_2 = PAM_CAM_Layer(channels[1] * self.count_list_2)
        self.skblock2_2 = ChannelAttention(channels[1] * self.count_list_2, channels[1] * 2, 16)
        self.conv_deco_2_3 = nn.Sequential(
            CNN_Block(channels[1] * self.count_list_2, channels[1], self.model_settings['to_decoder_2_conv']),
            CNN_Block(channels[1], channels[1], self.model_settings['to_decoder_2_conv']),
        )
        self.conv_deco_2_4 = nn.Sequential(
            CNN_Block(channels[1] * 2, channels[1], self.model_settings['to_decoder_2_conv']),
            CNN_Block(channels[1], channels[1], self.model_settings['to_decoder_2_conv']),
        )
        # </editor-fold>

        # <editor-fold desc=" decoder_1 ">
        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.attention1_0 = PAM_CAM_Layer(channels[0])
        self.attention1_1 = PAM_CAM_Layer(channels[0])
        self.attention1_1_b = PAM_CAM_Layer(channels[1])
        self.skblock1_1 = ChannelAttention(channels[0], channels[0], 16)
        self.skblock1_1_b = ChannelAttention(channels[1], channels[1], 16)

        self.AABlock_1_add = AttentionAggregationModule(channels[0], channels[0])
        self.AABlock_1_add_b = AttentionAggregationModule(channels[1], channels[1])

        self.AABlock_2_add = AttentionAggregationModule(channels[1], channels[1])
        self.AABlock_2_add_b = AttentionAggregationModule(channels[2], channels[2])

        self.AABlock_3_add = AttentionAggregationModule(channels[2], channels[2])
        self.AABlock_3_add_b = AttentionAggregationModule(channels[3], channels[3])

        self.AABlock_4_add = AttentionAggregationModule(channels[3], channels[3])
        self.AABlock_4_add_b = AttentionAggregationModule(channels[4], channels[4])

        self.conv_deco_1_1 = nn.Sequential(
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.conv_deco_1_2 = nn.Sequential(
            CNN_Block(channels[1], channels[0], self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.to_decoder_1_more_two_tensor = self.model_settings['to_decoder_1_more_two_tensor']
        self.to_decoder_1_more_two_tensor_arry = self.to_decoder_1_more_two_tensor.split("]--[")
        if self.to_decoder_1_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "full":
            self.count_list_1 = self.to_decoder_1_more_two_tensor_arry.count('con') + 1
        elif self.to_decoder_1_more_two_tensor[-3:] == "con" and self.model_settings['layer_num'] == "reduce_one":
            self.count_list_1 = self.to_decoder_1_more_two_tensor_arry.count('con')
        else:
            self.count_list_1 = self.to_decoder_1_more_two_tensor_arry.count('con') + 1

        self.attention1_2 = PAM_CAM_Layer(channels[0] * self.count_list_1)
        self.skblock1_2 = ChannelAttention(channels[0] * self.count_list_1, channels[0] * 2, 16)
        
        self.AABlock_1_1 = AttentionAggregationModule(channels[0] * self.count_list_1, channels[0])
        self.conv_deco_1_AAB = nn.Sequential(
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.conv_deco_1_AAB_list = nn.Sequential(
            CNN_Block(channels[0] * self.count_list_1, channels[0], self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.conv_deco_1_3 = nn.Sequential(
            CNN_Block(channels[0] * self.count_list_1, channels[0], self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )
        self.conv_deco_1_4 = nn.Sequential(
            CNN_Block(channels[0] * 2, channels[0], self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.conv_deco_1_4_add = nn.Sequential(
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        self.conv_deco_1_5 = nn.Sequential(
            CNN_Block(channels[0] * self.count_list_1, channels[0] * 2, self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0] * 2, channels[0], self.model_settings['to_decoder_1_conv']),
            CNN_Block(channels[0], channels[0], self.model_settings['to_decoder_1_conv']),
        )

        # </editor-fold>)
        self.AABlock = AttentionAggregationModule(channels[0] * 4, channels[0])
        self.AABlock_redu = AttentionAggregationModule(channels[0] * 3, channels[0])

        self.skblock0_1 = ChannelAttention(channels[0] * 4, channels[0] * 1, 16)
        self.skblock0_2 = ChannelAttention(channels[0] * 3, channels[0] * 1, 16)

        self.attention0_0 = PAM_CAM_Layer(channels[0])
        self.conv_deco_0_0 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)
        self.finaldeconv1 = nn.ConvTranspose2d(channels[0], channels_end, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(channels_end, channels_end, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(channels_end, self.class_num, 3, padding=1)

    def forward(self, x):
        # <editor-fold desc=" encoder ">
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        # x1 = self.firstmaxpool(x1)
        conv1 = self.conv1(x1)
        conv12 = self.conv12(conv1)
        conv13 = self.conv13(conv12)
        if self.model_settings['layer_num'] == "full":
            conv14 = self.conv14(conv13)
        conv2 = self.conv2(conv1)
        conv23 = self.conv23(conv2)
        if self.model_settings['layer_num'] == "full":
            conv24 = self.conv24(conv23)
        conv3 = self.conv3(conv2)
        if self.model_settings['layer_num'] == "full":
            conv34 = self.conv34(conv3)
            conv4 = self.conv4(conv3)
        # </editor-fold>

        if self.model_settings['layer_num'] == "reduce_one":
            if (self.model_settings['encoder_2_decoder_5'] == "att_pa"):
                conv5 = self.conv5_redu(conv3)
                conv5 = self.attention5_to_5_redu(conv5)
                conv5 = self.conv5_1_redu(conv5)
            elif self.model_settings['encoder_2_decoder_5'] == "skip_att":
                conv5 = self.conv5_redu(conv3)

        elif self.model_settings['layer_num'] == "full":
            if (self.model_settings['encoder_2_decoder_5'] == "att_pa"):
                conv5 = self.conv5(conv4)
                conv5 = self.attention5_to_5(conv5)
                conv5 = self.conv5_1(conv5)
            elif self.model_settings['encoder_2_decoder_5'] == "skip_att":
                conv5 = self.conv5(conv4)
        # </editor-fold>

        # <editor-fold desc=" decoder_4 ">
        if self.model_settings['layer_num'] == "full":
            deconv4 = self.deconv4(conv5)
            deconv43 = self.deconv43(deconv4)
            deconv42 = self.deconv42(deconv43)
            deconv41 = self.deconv41(deconv42)
            if (self.model_settings['to_decoder_4'] == "single"):
                if (self.model_settings['to_decoder_4_sin_att'] == "att"):
                    deconv4 = self.attention4_0(deconv4)

            elif (self.model_settings['to_decoder_4'] == "both"):
                if (self.model_settings['to_decoder_4_both_position'] == "after"):
                    if (self.model_settings['to_decoder_4_both_Connection_mode'] == "addition"):
                        if (self.model_settings['to_decoder_4_both_Connection_tensor'] == "one"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "two"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "three"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "four"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.conv_deco_4_1(deconv4)

                    elif (self.model_settings['to_decoder_4_both_Connection_mode'] == "concatence"):
                        if (self.model_settings['to_decoder_4_both_Connection_tensor'] == "one"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.attention4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.skblock4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.AABlock_4_add_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "two"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.attention4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.skblock4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.AABlock_4_add_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "three"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.attention4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.skblock4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.AABlock_4_add_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "four"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.attention4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.skblock4_1_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.AABlock_4_add_b(deconv4)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                elif (self.model_settings['to_decoder_4_both_position'] == "behind"):
                    if (self.model_settings['to_decoder_4_both_Connection_mode'] == "addition"):
                        if (self.model_settings['to_decoder_4_both_Connection_tensor'] == "one"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv14)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "two"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv24)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "three"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv34)
                                deconv4 = self.conv_deco_4_1(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "four"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.conv_deco_4_1(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                                deconv4 = torch.add(deconv4, conv4)
                                deconv4 = self.conv_deco_4_1(deconv4)

                    elif (self.model_settings['to_decoder_4_both_Connection_mode'] == "concatence"):
                        if (self.model_settings['to_decoder_4_both_Connection_tensor'] == "one"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv14), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "two"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv24), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "three"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv34), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

                        elif (self.model_settings['to_decoder_4_both_Connection_tensor'] == "four"):
                            if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                                deconv4 = self.attention4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                                deconv4 = self.skblock4_1(deconv4)
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                                deconv4 = self.AABlock_4_add(deconv4)
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)
                            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                                deconv4 = torch.cat((deconv4, conv4), 1)
                                deconv4 = self.conv_deco_4_2(deconv4)

            elif (self.model_settings['to_decoder_4'] == "more_two"):
                list_tensor = []
                list_tensor.append(deconv4)
                if (self.to_decoder_4_more_two_tensor_arry[3] == "con"):
                    list_tensor.append(conv4)
                if (self.to_decoder_4_more_two_tensor_arry[2] == "con"):
                    list_tensor.append(conv34)
                if (self.to_decoder_4_more_two_tensor_arry[1] == "con"):
                    list_tensor.append(conv24)
                if (self.to_decoder_4_more_two_tensor_arry[0] == "con"):
                    list_tensor.append(conv14)
                deconv4 = torch.cat(list_tensor, 1)
                if (self.model_settings['to_decoder_4_both_Connection_att'] == "PAM_CAM"):
                    deconv4 = self.attention4_2(deconv4)
                    deconv4 = self.conv_deco_4_3(deconv4)
                elif (self.model_settings['to_decoder_4_both_Connection_att'] == "Channel"):
                    deconv4 = self.skblock4_2(deconv4)
                    deconv4 = self.conv_deco_4_4(deconv4)
                elif (self.model_settings['to_decoder_4_both_Connection_att'] == "no_att"):
                    deconv4 = self.conv_deco_4_4_no_att(deconv4)
                elif (self.model_settings['to_decoder_4_both_Connection_att'] == "att_aam"):
                    deconv4 = self.AABlock_deco_4_4_add(deconv4)
                    deconv4 = self.conv_deco_4_4(deconv4)
        # </editor-fold>

        # <editor-fold desc=" decoder_3 ">
        if self.model_settings['layer_num'] == "full":
            deconv3 = self.deconv3(deconv4)
            deconv32 = self.deconv32(deconv3)
            deconv31 = self.deconv31(deconv32)
        elif self.model_settings['layer_num'] == "reduce_one":
            deconv3 = self.deconv3(conv5)
            deconv32 = self.deconv32(deconv3)
            deconv31 = self.deconv31(deconv32)

        if (self.model_settings['to_decoder_3'] == "single"):
            if (self.model_settings['to_decoder_3_sin_att'] == "att"):
                deconv3 = self.attention3_0(deconv3)

        elif (self.model_settings['to_decoder_3'] == "both"):
            if (self.model_settings['to_decoder_3_both_position'] == "after"):
                if (self.model_settings['to_decoder_3_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_3_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            print(deconv3.shape)
                            print(conv3.shape)
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.conv_deco_3_1(deconv3)

                elif (self.model_settings['to_decoder_3_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_3_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.attention3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.skblock3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.AABlock_3_add_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.attention3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.skblock3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.AABlock_3_add_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.attention3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.skblock3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.AABlock_3_add_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.attention3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.skblock3_1_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.AABlock_3_add_b(deconv3)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

            elif (self.model_settings['to_decoder_3_both_position'] == "behind"):
                if (self.model_settings['to_decoder_3_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_3_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv13)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv23)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, conv3)
                            deconv3 = self.conv_deco_3_1(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.conv_deco_3_1(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.add(deconv3, deconv43)
                            deconv3 = self.conv_deco_3_1(deconv3)

                elif (self.model_settings['to_decoder_3_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_3_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv13), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv23), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, conv3), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

                    elif (self.model_settings['to_decoder_3_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                            deconv3 = self.attention3_1(deconv3)
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                            deconv3 = self.skblock3_1(deconv3)
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                            deconv3 = self.AABlock_3_add(deconv3)
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)
                        elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                            deconv3 = torch.cat((deconv3, deconv43), 1)
                            deconv3 = self.conv_deco_3_2(deconv3)

        elif (self.model_settings['to_decoder_3'] == "more_two"):
            list_tensor = []
            list_tensor.append(deconv3)
            if (self.to_decoder_3_more_two_tensor_arry[3] == "con" and self.model_settings['layer_num'] == "full"):
                list_tensor.append(deconv43)
            if (self.to_decoder_3_more_two_tensor_arry[2] == "con"):
                list_tensor.append(conv3)
            if (self.to_decoder_3_more_two_tensor_arry[1] == "con"):
                list_tensor.append(conv23)
            if (self.to_decoder_3_more_two_tensor_arry[0] == "con"):
                list_tensor.append(conv13)
            deconv3 = torch.cat(list_tensor, 1)
            if (self.model_settings['to_decoder_3_both_Connection_att'] == "PAM_CAM"):
                deconv3 = self.attention3_2(deconv3)
                deconv3 = self.conv_deco_3_3(deconv3)
            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "Channel"):
                deconv3 = self.skblock3_2(deconv3)
                deconv3 = self.conv_deco_3_4(deconv3)
            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "no_att"):
                deconv3 = self.conv_deco_3_3(deconv3)
            elif (self.model_settings['to_decoder_3_both_Connection_att'] == "att_aam"):
                deconv3 = self.AABlock_deco_3_4(deconv3)
                deconv3 = self.conv_deco_3_4(deconv3)
        # </editor-fold>

        # <editor-fold desc=" decoder_2 ">
        deconv2 = self.deconv2(deconv3)
        deconv21 = self.deconv21(deconv2)
        if (self.model_settings['to_decoder_2'] == "single"):
            if (self.model_settings['to_decoder_2_sin_att'] == "att"):
                deconv2 = self.attention2_0(deconv2)
        elif (self.model_settings['to_decoder_2'] == "both"):
            if (self.model_settings['to_decoder_2_both_position'] == "after"):
                if (self.model_settings['to_decoder_2_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_2_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.conv_deco_2_1(deconv2)

                elif (self.model_settings['to_decoder_2_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_2_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.attention2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.skblock2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.AABlock_2_add_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.attention2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.skblock2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.AABlock_2_add_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.attention2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.skblock2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.AABlock_2_add_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.attention2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.skblock2_1_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.AABlock_2_add_b(deconv2)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

            elif (self.model_settings['to_decoder_2_both_position'] == "behind"):
                if (self.model_settings['to_decoder_2_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_2_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, conv12)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, conv2)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, deconv32)
                            deconv2 = self.conv_deco_2_1(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.conv_deco_2_1(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.add(deconv2, deconv42)
                            deconv2 = self.conv_deco_2_1(deconv2)

                elif (self.model_settings['to_decoder_2_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_2_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, conv12), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, conv2), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            # print(deconv2.shape)
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, deconv32), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)

                    elif (self.model_settings['to_decoder_2_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                            deconv2 = self.attention2_1(deconv2)
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                            deconv2 = self.skblock2_1(deconv2)
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                            deconv2 = self.AABlock_2_add(deconv2)
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
                        elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                            deconv2 = torch.cat((deconv2, deconv42), 1)
                            deconv2 = self.conv_deco_2_2(deconv2)
        elif (self.model_settings['to_decoder_2'] == "more_two"):
            list_tensor = []
            list_tensor.append(deconv2)
            if (self.to_decoder_2_more_two_tensor_arry[3] == "con" and self.model_settings['layer_num'] == "full"):
                list_tensor.append(deconv42)
            if (self.to_decoder_2_more_two_tensor_arry[2] == "con"):
                list_tensor.append(deconv32)
            if (self.to_decoder_2_more_two_tensor_arry[1] == "con"):
                list_tensor.append(conv2)
            if (self.to_decoder_2_more_two_tensor_arry[0] == "con"):
                list_tensor.append(conv12)
            deconv2 = torch.cat(list_tensor, 1)
            if (self.model_settings['to_decoder_2_both_Connection_att'] == "PAM_CAM"):
                deconv2 = self.attention2_2(deconv2)
                deconv2 = self.conv_deco_2_3(deconv2)
            elif (self.model_settings['to_decoder_2_both_Connection_att'] == "Channel"):
                deconv2 = self.skblock2_2(deconv2)
                deconv2 = self.conv_deco_2_4(deconv2)
            elif (self.model_settings['to_decoder_2_both_Connection_att'] == "no_att"):
                deconv2 = self.conv_deco_2_3(deconv2)
            elif (self.model_settings['to_decoder_2_both_Connection_att'] == "att_aam"):
                deconv2 = self.AABlock_deco_2_4(deconv2)
                deconv2 = self.conv_deco_2_4(deconv2)
        # </editor-fold>

        # <editor-fold desc=" decoder_1 ">
        deconv1 = self.deconv1(deconv2)
        if (self.model_settings['to_decoder_1'] == "single"):
            if (self.model_settings['to_decoder_1_sin_att'] == "att"):
                deconv1 = self.attention1_0(deconv1)
        elif (self.model_settings['to_decoder_1'] == "both"):
            if (self.model_settings['to_decoder_1_both_position'] == "after"):
                if (self.model_settings['to_decoder_1_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_1_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.conv_deco_1_1(deconv1)

                elif (self.model_settings['to_decoder_1_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_1_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.attention1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.skblock1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.AABlock_1_add_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.attention1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.skblock1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.AABlock_1_add_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.attention1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.skblock1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.AABlock_1_add_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.attention1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.skblock1_1_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.AABlock_1_add_b(deconv1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

            elif (self.model_settings['to_decoder_1_both_position'] == "behind"):
                if (self.model_settings['to_decoder_1_both_Connection_mode'] == "addition"):
                    if (self.model_settings['to_decoder_1_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, conv1)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv21)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv31)
                            deconv1 = self.conv_deco_1_1(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.conv_deco_1_1(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.add(deconv1, deconv41)
                            deconv1 = self.conv_deco_1_1(deconv1)

                elif (self.model_settings['to_decoder_1_both_Connection_mode'] == "concatence"):
                    if (self.model_settings['to_decoder_1_both_Connection_tensor'] == "one"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, conv1), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "two"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv21), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "three"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv31), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)

                    elif (self.model_settings['to_decoder_1_both_Connection_tensor'] == "four" and self.model_settings[
                        'layer_num'] == "full"):
                        if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                            deconv1 = self.attention1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                            deconv1 = self.skblock1_1(deconv1)
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                            deconv1 = self.AABlock_1_add(deconv1)
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
                        elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                            deconv1 = torch.cat((deconv1, deconv41), 1)
                            deconv1 = self.conv_deco_1_2(deconv1)
        elif (self.model_settings['to_decoder_1'] == "more_two"):
            list_tensor = []
            list_tensor.append(deconv1)
            if (self.to_decoder_1_more_two_tensor_arry[3] == "con" and self.model_settings['layer_num'] == "full"):
                list_tensor.append(deconv41)
            if (self.to_decoder_1_more_two_tensor_arry[2] == "con"):
                list_tensor.append(deconv31)
            if (self.to_decoder_1_more_two_tensor_arry[1] == "con"):
                list_tensor.append(deconv21)
            if (self.to_decoder_1_more_two_tensor_arry[0] == "con"):
                list_tensor.append(conv1)
            deconv1 = torch.cat(list_tensor, 1)
            if (self.model_settings['to_decoder_1_both_Connection_att'] == "PAM_CAM"):
                deconv1 = self.attention1_2(deconv1)
                deconv1 = self.conv_deco_1_AAB_list(deconv1)
            elif (self.model_settings['to_decoder_1_both_Connection_att'] == "Channel"):
                deconv1 = self.skblock1_2(deconv1)
                deconv1 = self.conv_deco_1_4(deconv1)
            elif (self.model_settings['to_decoder_1_both_Connection_att'] == "att_aam"):
                deconv1 = self.AABlock_1_1(deconv1)
                deconv1 = self.conv_deco_1_4_add(deconv1)
            elif (self.model_settings['to_decoder_1_both_Connection_att'] == "no_att"):
                deconv1 = self.conv_deco_1_5(deconv1)
        # </editor-fold>

        # <editor-fold desc=" output ">
        if (self.model_settings['decoder_output_att'] == "PAM_CAM"):
            output = self.attention0_0(deconv1)
            out = self.finaldeconv1(output)
            out = self.finalrelu1(out)
            out = self.finalconv2(out)
            out = self.finalrelu2(out)
            output = self.finalconv3(out)

        elif (self.model_settings['decoder_output_att'] == "no_att"):
            out = self.finaldeconv1(deconv1)
            out = self.finalrelu1(out)
            out = self.finalconv2(out)
            out = self.finalrelu2(out)
            output = self.finalconv3(out)
        # </editor-fold>
        return output
