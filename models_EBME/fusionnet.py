import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from EBME.utils.common_op import conv2, conv3, conv4, deconv, deconv2, deconv3
from EBME.softsplat import softsplat
import math
import torch.distributions as D
from torch.nn import Conv2d


def downsample_image(img, mask):
    """ down-sample the image [H*2, W*2, 3] -> [H, W, 2] using convex combination """
    N, _, H, W = img.shape
    mask = mask.view(N, 1, 25, H // 2, W // 2)
    mask = torch.softmax(mask, dim=2)

    down_img = F.unfold(img, [5,5], stride=2, padding=2)
    down_img = down_img.view(N, 3, 25, H // 2, W // 2)

    down_img = torch.sum(mask * down_img, dim=2)
    return down_img


class ContextNet(nn.Module):
    def __init__(self, c=16):
        c = c
        super(ContextNet, self).__init__()
        self.conv1 = conv2(3, c)
        self.conv2 = conv2(c, 2*c)
        self.conv3 = conv2(2*c, 4*c)
        self.conv4 = conv2(4*c, 8*c)

    def forward(self, img, flow, ):
        feat_pyramid = []

        # calculate forward-warped feature pyramid
        feat = img
        feat = self.conv1(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat1 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat1)

        feat = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat2 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat2)

        feat = self.conv3(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat3 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat3)

        feat = self.conv4(feat)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        warped_feat4 = softsplat.FunctionSoftsplat(
                tenInput=feat, tenFlow=flow,
                tenMetric=None, strType='average')
        feat_pyramid.append(warped_feat4)

        return feat_pyramid



class FusionNet(nn.Module):
    def __init__(self, args):
        super(FusionNet, self).__init__()
        c = 16
        self.high_synthesis = args.high_synthesis if "high_synthesis" in args else False
        self.contextnet = ContextNet()
        self.down1 = conv4(16, 2*c)
        self.down2 = conv2(4*c, 4*c)
        self.down3 = conv2(8*c, 8*c)
        self.down4 = conv2(16*c, 16*c)
        self.up1 = deconv(32*c, 8*c)
        self.up2 = deconv(16*c, 4*c)
        self.up3 = deconv(8*c, 2*c)
        self.up4 = deconv3(4*c, c)
        self.refine_pred = nn.Conv2d(c, 4, 3, 1, 1)
        if self.high_synthesis:
            self.downsample_mask = nn.Sequential(
                nn.Conv2d(c, 2*c, 5, 2, 2),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 2*c, 3, 1, 1),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 25, 1, padding=0))

        # fix the paramters if needed
        if ("fix_pretrain" in args) and (args.fix_pretrain):
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # upsample input images and estimated bi_flow, if using the
        # "high_synthesis" setting.
        if self.high_synthesis:
            img0 = F.interpolate(input=img0, scale_factor=2.0, mode="bilinear", align_corners=False)
            img1 = F.interpolate(input=img1, scale_factor=2.0, mode="bilinear", align_corners=False)
            bi_flow = F.interpolate(
                    input=bi_flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0

        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        c0 = self.contextnet(img0, flow_0t)
        c1 = self.contextnet(img1, flow_1t)

        # feature extraction by u-net
        s0 = self.down1(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1))
        # s0 = self.down1(torch.cat(((img1 - warped_img0), (img0 - warped_img1), (img1 - img0), flow_0t_1t), 1))
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))

        # prediction
        refine = self.refine_pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask = torch.sigmoid(refine[:, 3:4])
        merged_img = warped_img0 * refine_mask + warped_img1 * (1 - refine_mask)
        interp_img = merged_img + refine_res
        interp_img = torch.clamp(interp_img, 0, 1)  # removed by xmfeng

        # convex down-sampling, if using "high_synthesis" setting.
        if self.high_synthesis:
            downsample_mask = self.downsample_mask(x)
            interp_img = downsample_image(interp_img, downsample_mask)

        return interp_img


class FusionNetTVRNbaseline(nn.Module):
    def __init__(self, args):
        super(FusionNetTVRNbaseline, self).__init__()
        c = 4
        self.high_synthesis = args.high_synthesis if "high_synthesis" in args else False
        self.contextnet = ContextNet(c=c)
        self.down1 = conv4(7, 2*c)
        self.down2 = conv2(4*c, 4*c)
        self.down3 = conv2(8*c, 8*c)
        self.down4 = conv2(16*c, 16*c)
        self.up1 = deconv(32*c, 8*c)
        self.up2 = deconv(16*c, 4*c)
        self.up3 = deconv(8*c, 2*c)
        self.up4 = deconv3(4*c, c)
        self.refine_pred = nn.Conv2d(c, 4, 3, 1, 1, padding_mode='replicate')
        self.init_res = nn.Conv2d(3, 3, 3, 1, 1, padding_mode='replicate')
        if self.high_synthesis:
            self.downsample_mask = nn.Sequential(
                nn.Conv2d(c, 2*c, 5, 2, 2),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 2*c, 3, 1, 1),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 25, 1, padding=0))

        # fix the paramters if needed
        if ("fix_pretrain" in args) and (args.fix_pretrain):
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # upsample input images and estimated bi_flow, if using the
        # "high_synthesis" setting.
        # if self.high_synthesis:
        #     img0 = F.interpolate(input=img0, scale_factor=2.0, mode="bilinear", align_corners=False)
        #     img1 = F.interpolate(input=img1, scale_factor=2.0, mode="bilinear", align_corners=False)
        #     bi_flow = F.interpolate(
        #             input=bi_flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0

        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        
        # return torch.cat([(warped_img1 - warped_img0), img1 - img0], dim=1), warped_img1, warped_img0

        c0 = self.contextnet((warped_img0 - img1), flow_0t)
        c1 = self.contextnet((warped_img1 - img0), flow_1t)

        # c0 = self.contextnet((img0), flow_0t)
        # c1 = self.contextnet((img1), flow_1t)

        coarse_res = self.init_res(warped_img1 - warped_img0)

        # feature extraction by u-net
        # s0 = self.down1(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1))
        s0 = self.down1(torch.cat((coarse_res, flow_0t_1t), 1))
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))

        # prediction
        refine = self.refine_pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask = torch.sigmoid(refine[:, 3:4])
        # merged_img = (warped_img1 - img0) * refine_mask + (img0 - warped_img1) * (1 - refine_mask)
        pred_hf = coarse_res * refine_mask + refine_res * (1 - refine_mask)
        # interp_img = torch.clamp(interp_img, 0, 1)  # removed by xmfeng

        # convex down-sampling, if using "high_synthesis" setting.
        if self.high_synthesis:
            downsample_mask = self.downsample_mask(x)
            interp_img = downsample_image(interp_img, downsample_mask)

        return pred_hf, warped_img1, warped_img0


class PFNLWithBlocks_wo_context(nn.Module):
    def __init__(self, args):
        super(PFNLWithBlocks_wo_context, self).__init__()
        c = 4
        self.high_synthesis = args.high_synthesis if "high_synthesis" in args else False
        # self.contextnet = ContextNet(c=c)
        self.down1 = conv4(7, 2*c)
        self.down2 = conv2(4*c, 4*c)
        self.down3 = conv2(8*c, 8*c)
        self.down4 = conv2(16*c, 16*c)
        self.up1 = deconv(32*c, 8*c)
        self.up2 = deconv(16*c, 4*c)
        self.up3 = deconv(8*c, 2*c)
        self.up4 = deconv3(4*c, c)
        self.refine_pred = nn.Conv2d(c, 4, 3, 1, 1, padding_mode='replicate')
        self.init_res = nn.Conv2d(3, 3, 3, 1, 1, padding_mode='replicate')
        if self.high_synthesis:
            self.downsample_mask = nn.Sequential(
                nn.Conv2d(c, 2*c, 5, 2, 2),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 2*c, 3, 1, 1),
                nn.PReLU(2*c),
                nn.Conv2d(2*c, 25, 1, padding=0))

        # fix the paramters if needed
        if ("fix_pretrain" in args) and (args.fix_pretrain):
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        
        # return torch.cat([(warped_img1 - warped_img0), img1 - img0], dim=1), warped_img1, warped_img0

        # c0 = self.contextnet((warped_img0 - img1), flow_0t)
        # c1 = self.contextnet((warped_img1 - img0), flow_1t)

        c0 = [torch.zeros_like(img1[:,0:1,::2,::2]).repeat(1,4,1,1), 
              torch.zeros_like(img1[:,0:1,::4,::4]).repeat(1,8,1,1), 
              torch.zeros_like(img1[:,0:1,::8,::8]).repeat(1,16,1,1), 
              torch.zeros_like(img1[:,0:1,::16,::16]).repeat(1,32,1,1)]
        c1 = [torch.zeros_like(img1[:,0:1,::2,::2]).repeat(1,4,1,1), 
              torch.zeros_like(img1[:,0:1,::4,::4]).repeat(1,8,1,1), 
              torch.zeros_like(img1[:,0:1,::8,::8]).repeat(1,16,1,1), 
              torch.zeros_like(img1[:,0:1,::16,::16]).repeat(1,32,1,1)]
        
        coarse_res = self.init_res(warped_img1 - warped_img0)

        # feature extraction by u-net
        # s0 = self.down1(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1))
        s0 = self.down1(torch.cat((coarse_res, flow_0t_1t), 1))
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))

        # prediction
        refine = self.refine_pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask = torch.sigmoid(refine[:, 3:4])
        # merged_img = (warped_img1 - img0) * refine_mask + (img0 - warped_img1) * (1 - refine_mask)
        pred_hf = coarse_res * refine_mask + refine_res * (1 - refine_mask)
        # interp_img = torch.clamp(interp_img, 0, 1)  # removed by xmfeng

        # convex down-sampling, if using "high_synthesis" setting.
        if self.high_synthesis:
            downsample_mask = self.downsample_mask(x)
            interp_img = downsample_image(interp_img, downsample_mask)

        return pred_hf, warped_img1, warped_img0



# class PFNLWithBlocks_wo_context(nn.Module):
#     def __init__(self, gop_size=7, input_channels=3,  mid_channels=48):
#         super(PFNLWithBlocks_wo_context, self).__init__()
#         self.num_frames = math.ceil(gop_size / 2)
#         self.num_block = 5

#         # self.rfrb_blocks = nn.ModuleList([
#         #     PFNL_wo_context(gop_size=gop_size, input_channels=input_channels, 
#         #          context_channels=context_channels, mid_channels=mid_channels)
#         #     for _ in range(self.num_block)
#         # ])

#         self.rfrb_blocks = nn.ModuleList([
#             PFNL_wo_context_HF_recon(len=self.num_frames, input_channels=input_channels,  mid_channels=mid_channels)
#             for _ in range(self.num_block)
#         ])

#     def forward(self, x):
#         for rfrb_block in self.rfrb_blocks:
#             x = x + rfrb_block(x)
#         return x

class FusionNetTVRN(nn.Module):
    def __init__(self, args):
        super(FusionNetTVRN, self).__init__()
        # c = 4
        # self.high_synthesis = args.high_synthesis if "high_synthesis" in args else False
        # self.contextnet = ContextNet(c=c)
        # self.down1 = conv4(7, 2*c)
        # self.down2 = conv2(4*c, 4*c)
        # self.down3 = conv2(8*c, 8*c)
        # self.down4 = conv2(16*c, 16*c)
        # self.up1 = deconv(32*c, 8*c)
        # self.up2 = deconv(16*c, 4*c)
        # self.up3 = deconv(8*c, 2*c)
        # self.up4 = deconv3(4*c, c)
        # self.refine_pred = nn.Conv2d(c, 4, 3, 1, 1)
        # self.init_res = nn.Conv2d(3, 3, 3, 1, 1)
        # if self.high_synthesis:
        #     self.downsample_mask = nn.Sequential(
        #         nn.Conv2d(c, 2*c, 5, 2, 2),
        #         nn.PReLU(2*c),
        #         nn.Conv2d(2*c, 2*c, 3, 1, 1),
        #         nn.PReLU(2*c),
        #         nn.Conv2d(2*c, 25, 1, padding=0))

        # # fix the paramters if needed
        # if ("fix_pretrain" in args) and (args.fix_pretrain):
        #     for p in self.parameters():
        #         p.requires_grad = False


    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # upsample input images and estimated bi_flow, if using the
        # "high_synthesis" setting.
        # if self.high_synthesis:
        #     img0 = F.interpolate(input=img0, scale_factor=2.0, mode="bilinear", align_corners=False)
        #     img1 = F.interpolate(input=img1, scale_factor=2.0, mode="bilinear", align_corners=False)
        #     bi_flow = F.interpolate(
        #             input=bi_flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0

        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        
        return torch.cat([(warped_img1 - warped_img0), img1 - img0], dim=1), warped_img1, warped_img0

        c0 = self.contextnet((warped_img0 - img1), flow_0t)
        c1 = self.contextnet((warped_img1 - img0), flow_1t)

        coarse_res = self.init_res(warped_img1 - warped_img0)

        # feature extraction by u-net
        # s0 = self.down1(torch.cat((warped_img0, warped_img1, img0, img1, flow_0t_1t), 1))
        s0 = self.down1(torch.cat((coarse_res, flow_0t_1t), 1))
        s1 = self.down2(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down3(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down4(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up2(torch.cat((x, s2), 1))
        x = self.up3(torch.cat((x, s1), 1))
        x = self.up4(torch.cat((x, s0), 1))

        # prediction
        refine = self.refine_pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask = torch.sigmoid(refine[:, 3:4])
        # merged_img = (warped_img1 - img0) * refine_mask + (img0 - warped_img1) * (1 - refine_mask)
        pred_hf = coarse_res * refine_mask + refine_res * (1 - refine_mask)
        # interp_img = torch.clamp(interp_img, 0, 1)  # removed by xmfeng

        # convex down-sampling, if using "high_synthesis" setting.
        if self.high_synthesis:
            downsample_mask = self.downsample_mask(x)
            interp_img = downsample_image(interp_img, downsample_mask)

        return pred_hf, warped_img1, warped_img0


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True, t=None):
        super(D2DTInput, self).__init__()
        self.t = t
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            initialize_weights(self.conv5, 0)

    def forward(self, x):
        # b c t h w
        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.t)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return rearrange(x5, 'b c t h w -> (b t) c h w')


class GlobalAgg(nn.Module):
    def __init__(self,c, t=None):
        super(GlobalAgg, self).__init__()
        self.fc = nn.Linear(32*32,1)
        self.proj1 = nn.Conv2d(c,c,1,1,0)
        self.proj2 = nn.Linear(c,c)
        self.proj3 = nn.Linear(c,c)
        self.t = t

    def forward(self, x):
        ### 64 w h 
        TEMP_LEN = self.t
        x_proj1 = self.proj1(x)

        B,C,H,W = x.size()
        x_down_sample = F.adaptive_avg_pool2d(x,output_size = (32,32))
        x_down_sample = x_down_sample.reshape(B,C,32*32)
        x_down_sample = self.fc(x_down_sample).squeeze()
        x_down_sample_video = x_down_sample.reshape(B//TEMP_LEN,TEMP_LEN,C)
        x_down_sample_video_proj2 = self.proj2(x_down_sample_video)
        x_down_sample_video_proj3 = self.proj3(x_down_sample_video)
        temporal_weight_matrix = torch.matmul(x_down_sample_video_proj2,x_down_sample_video_proj3.transpose(1,2))
        #### T * T
        temporal_weight_matrix = F.softmax(temporal_weight_matrix/C, dim=-1)

        x_proj1 = x_proj1.reshape(B//TEMP_LEN,TEMP_LEN,C,H,W)
        x_proj1 = x_proj1.permute(0,2,3,4,1).reshape(B//TEMP_LEN,C*H*W,TEMP_LEN)
        weighted_feature = torch.matmul(x_proj1,temporal_weight_matrix) ## b (chw) t

        return x + weighted_feature.reshape(B//TEMP_LEN,C,H,W,TEMP_LEN).\
            permute(0,4,1,2,3).reshape(B,C,H,W)



class GaussianMixture(nn.Module):
    def __init__(self, gmm_components=5):
        super(GaussianMixture, self).__init__()
        self.gmm_para = nn.Parameter(0.0001 * torch.randn(3, gmm_components)) # init
    
    def refresh(self):
        self.pis, self.mus, self.logvars = self.gmm_para

    def build(self):
        self.refresh()
        std = torch.exp(0.5*torch.clamp(self.logvars, -20, 20))
        weights = self.pis.softmax(dim=0)
        mix = D.Categorical(weights)
        comp = D.Normal(self.mus, std)
        self.gmm = D.MixtureSameFamily(mix, comp)
        
    def sample(self, zshape):
        self.build()
        assert (
            self.gmm.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.gmm.mixture_distribution._param.repeat(*zshape, 1)
        comp = nn.functional.gumbel_softmax(weights.log(), hard=True)
        samples = self.gmm.component_distribution.rsample(zshape)
        return (comp * samples).sum(dim=-1)

class FusionNetTVRNGMM(nn.Module):
    def __init__(self, args):
        super(FusionNetTVRNGMM, self).__init__()

    def forward(self, img0, img1, bi_flow, time_period=0.5, profile_time=False):
        # input features for sythesis network: original images, warped images, warped features, and flow_0t_1t
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
        warped_img0 = softsplat.FunctionSoftsplat(
                tenInput=img0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_img1 = softsplat.FunctionSoftsplat(
                tenInput=img1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        
        # 使用selfC的概率预测模块
        return torch.cat([(warped_img1 - warped_img0), img1 - img0], dim=1), warped_img1, warped_img0


class HfReconGMM(nn.Module):
    def __init__(self, args):
        super(HfReconGMM, self).__init__()
        c = 24
        gc = 12
        self.stp_blk_num = 2
        self.local_m1 = D2DTInput(6,c,gc =gc, INN_init=False, t=math.floor(args['gop_size'] / 2))
        self.local_m2 = D2DTInput(c,c,gc =gc, INN_init=False, t=math.floor(args['gop_size'] / 2))
        self.global_m1 = GlobalAgg(c, t=math.floor(args['gop_size'] / 2))
        self.global_m2 = GlobalAgg(c, t=math.floor(args['gop_size'] / 2))
        self.other_stp_modules = []
        for i in range(self.stp_blk_num):
            self.other_stp_modules +=[D2DTInput(c,c,gc =gc,INN_init=False, t=math.floor(args['gop_size'] / 2))]
            self.other_stp_modules +=[GlobalAgg(c, t=math.floor(args['gop_size'] / 2))]
        self.other_stp_modules = nn.Sequential(*self.other_stp_modules)
        MLP_dim = c
        self.hf_dim = 3 
        self.K = 5
        self.tail = [
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(c, MLP_dim*2, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(MLP_dim*2, MLP_dim*4, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(MLP_dim*4, self.hf_dim * self.K*3, 1, 1, 0, bias=True)
        ]
        self.tail = nn.Sequential(*self.tail)

        # self.gmm = GaussianMixture(5)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.cuda.FloatTensor(std.size()).fill_(0.0)
        eps.normal_()
        x=eps.mul(std).add_(mu)
        return x

    def forward(self, hf_init):
        # b c t h w
        b, c, t, h, w = hf_init.shape
        hf_init = rearrange(hf_init, 'b c t h w -> (b t) c h w')
        temp = self.local_m1(hf_init)  # b c t h w
        temp = self.global_m1(temp)
        temp = self.local_m2(temp)
        temp = self.global_m2(temp)
        temp = self.other_stp_modules(temp)
        temp = rearrange(temp, '(b t) c h w -> b c t h w', b=b)
        self.parameters = self.tail(temp)

        out_param = self.parameters

        b,c,t,h,w = out_param.size()
        out_param = out_param.reshape(b,self.hf_dim,self.K,3,t,h,w)
        pi = F.softmax(out_param[:,:,:,0],dim=1)
        log_scale = torch.clamp(out_param[:,:,:,1],-7,7)
        mean = out_param[:,:,:,2]
        
        v=pi* self.reparametrize(mean, log_scale) 
        v = v.sum(2)

        pred_hf = v
        
        return pred_hf, pi, mean, log_scale



class HfReconSyn(nn.Module):
    def __init__(self, args):
        super(HfReconSyn, self).__init__()
        c = 24
        gc = 12
        self.stp_blk_num = 2
        self.local_m1 = D2DTInput(6,c,gc =gc, INN_init=False, t=math.floor(args['gop_size'] / 2))
        self.local_m2 = D2DTInput(c,c,gc =gc, INN_init=False, t=math.floor(args['gop_size'] / 2))
        self.global_m1 = GlobalAgg(c, t=math.floor(args['gop_size'] / 2))
        self.global_m2 = GlobalAgg(c, t=math.floor(args['gop_size'] / 2))
        self.other_stp_modules = []
        for i in range(self.stp_blk_num):
            self.other_stp_modules +=[D2DTInput(c,c,gc =gc,INN_init=False, t=math.floor(args['gop_size'] / 2))]
            self.other_stp_modules +=[GlobalAgg(c, t=math.floor(args['gop_size'] / 2))]
        self.other_stp_modules = nn.Sequential(*self.other_stp_modules)
        MLP_dim = c
        self.hf_dim = 3 
        self.K = 5
        self.tail = [
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(c, MLP_dim*2, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(MLP_dim*2, MLP_dim*4, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(MLP_dim*4, 3, 1, 1, 0, bias=True)
        ]
        self.tail = nn.Sequential(*self.tail)


    def forward(self, hf_init):
        # b c t h w
        b, c, t, h, w = hf_init.shape
        hf_init = rearrange(hf_init, 'b c t h w -> (b t) c h w')
        temp = self.local_m1(hf_init)  # b c t h w
        temp = self.global_m1(temp)
        temp = self.local_m2(temp)
        temp = self.global_m2(temp)
        temp = self.other_stp_modules(temp)
        temp = rearrange(temp, '(b t) c h w -> b c t h w', b=b)
        pred_hf = self.tail(temp)

        return pred_hf


class PFNL(nn.Module):
    def __init__(self, gop_size=3, input_channels=3, context_channels=12, mid_channels=12):
        super(PFNL, self).__init__()
        self.num_frames = math.floor(gop_size / 2)

        self.conv0 = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.convmerge = nn.Conv2d(self.num_frames * mid_channels, mid_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(mid_channels * 3, 3, 3, 1, 1)

        self.conv0_context = nn.Conv2d(context_channels, mid_channels, 3, 1, 1)
        self.convmerge_context = nn.Conv2d(self.num_frames * mid_channels, mid_channels, 1, 1, 0)

    def forward(self, x, context):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_1 = self.conv0(x)
        x_1 = rearrange(x_1, '(b t) c h w -> b t c h w', b=b)
        x_2 = self.convmerge(rearrange(x_1, 'b t c h w -> b (t c) h w'))
        x_2 = repeat(x_2.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.num_frames)

        context = rearrange(context, 'b c t h w -> (b t) c h w')
        x_1_context = self.conv0_context(context)
        x_1_context = rearrange(x_1_context, '(b t) c h w -> b t c h w', b=b)
        x_2_context = self.convmerge(rearrange(x_1_context, 'b t c h w -> b (t c) h w'))
        x_2_context = repeat(x_2_context.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.num_frames)

        x_4 = torch.cat([x_1, x_2, x_2_context], dim=2)
        x_4 = rearrange(x_4, 'b t c h w -> (b t) c h w')
        x_4 = self.conv2(x_4)
        out = x + x_4
        return rearrange(out, '(b t) c h w -> b c t h w', b=b)


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super(Residual3DBlock, self).__init__()
        padding_modes = 'replicate'
        # padding_modes = 'zeros'
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_modes	)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_modes)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, padding_mode=padding_modes) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return (self.conv2(self.conv1(x)) + self.shortcut(x))
        # return (self.conv2(self.relu(self.conv1(x))) + self.shortcut(x))

class PFNL_wo_context_HF_recon(nn.Module):
    def __init__(self, len=3, input_channels=3, mid_channels=12):
        super(PFNL_wo_context_HF_recon, self).__init__()
        self.num_frames = len
        self.res3d_1 = Residual3DBlock(input_channels, mid_channels)
        self.res3d_2 = Residual3DBlock(mid_channels, mid_channels // len)
        self.res3d_3 = Residual3DBlock(mid_channels + mid_channels // len, mid_channels)

    def forward(self, x):
        x_1 = self.res3d_1(x)
        x_2 = self.res3d_2(x_1)
        x_4 = torch.cat([x_1, x_2], dim=1)
        x_4 = self.res3d_3(x_4)
        out = x + x_4
        return out

import models.modules.module_util as mutil

class DenseBlock3D(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5
    

class DenseBlock2D(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='replicate')
        # self.bn1 = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, padding_mode='replicate')
        # self.bn2 = nn.BatchNorm3d(out_channels)

        # 如果输入和输出通道数不同，使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                # nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += self.shortcut(residual)
        # out = self.relu(out)

        return out
    

    
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='replicate')
        # self.bn1 = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, padding_mode='replicate')
        # self.bn2 = nn.BatchNorm3d(out_channels)

        # 如果输入和输出通道数不同，使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                # nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += self.shortcut(residual)
        # out = self.relu(out)

        return out
    
class PFNLWithBlocks(nn.Module):
    def __init__(self, num_frames=7, input_channels=3, context_channels=3, mid_channels=12, num_blocks=3):
        super(PFNLWithBlocks, self).__init__()
        self.num_block = num_blocks
        # self.conv1 = DenseBlock(input_channels, mid_channels, gc=32)
        # tmp_list = [DenseBlock3D(input_channels, mid_channels, gc=16), ]
        # for _ in range(self.num_block - 2):
        #     tmp_list.append(DenseBlock3D(mid_channels, mid_channels, gc=16))
        # tmp_list.append(DenseBlock3D(mid_channels, input_channels, gc=16))

        # tmp_list = [DenseBlock2D(input_channels, mid_channels, gc=16), ]
        # for _ in range(self.num_block - 2):
        #     tmp_list.append(DenseBlock2D(mid_channels, mid_channels, gc=16))
        # tmp_list.append(DenseBlock2D(mid_channels, input_channels, gc=16))


        tmp_list = [ResidualBlock3D(input_channels, mid_channels, ), ]
        for _ in range(self.num_block - 2):
            tmp_list.append(ResidualBlock3D(mid_channels, mid_channels, ))
        tmp_list.append(ResidualBlock3D(mid_channels, input_channels, ))

        # self.head = nn.Conv2d(mid_channels + input_channels, input_channels, 1, 1, 0, padding_mode='replicate')

        # tmp_list = []
        # for _ in range(self.num_block):
        #     tmp_list.append(PFNL_wo_context_HF_recon(len=num_frames, input_channels=mid_channels,  mid_channels=mid_channels))
        # tmp_list.append(nn.Conv3d(mid_channels, input_channels, kernel_size=1, stride=1, padding=0, padding_mode='replicate'))
        self.rfrb_blocks_wo_context = nn.ModuleList(tmp_list)

    def forward(self, x, context=None):
        out = []
        b, c, t, h, w = x.shape
        # x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_init = x

        # x = self.conv1(x)
        for rfrb_block in self.rfrb_blocks_wo_context:
            x = rfrb_block(x)

        # x = self.head(torch.concat((x,  x_init), 1))
        # x = self.head(x)
        
        return [x, ]
        # return [rearrange(x, '(b t) c h w -> b c t h w', b=b), ]
            # out.append(rearrange(x, '(b t) c h w -> b c t h w', b=b))
        # return out
    

class PFNL_wo_context(nn.Module):
    def __init__(self, gop_size=3, input_channels=3, context_channels=12, mid_channels=12):
        super(PFNL_wo_context, self).__init__()
        self.num_frames = math.ceil(gop_size / 2)

        self.conv0 = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.convmerge = nn.Conv2d(self.num_frames * mid_channels, mid_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(mid_channels * 2, 3, 3, 1, 1)


    def forward(self, x,):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_1 = self.conv0(x)
        x_1 = rearrange(x_1, '(b t) c h w -> b t c h w', b=b)
        x_2 = self.convmerge(rearrange(x_1, 'b t c h w -> b (t c) h w'))
        x_2 = repeat(x_2.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.num_frames)

        x_4 = torch.cat([x_1, x_2], dim=2)
        x_4 = rearrange(x_4, 'b t c h w -> (b t) c h w')
        x_4 = self.conv2(x_4)
        out = x + x_4
        return rearrange(out, '(b t) c h w -> b c t h w', b=b)






class PFNL_old(nn.Module):
    def __init__(self, gop_size=3, input_channels=3, context_channels=12, mid_channels=12):
        super(PFNL_old, self).__init__()
        self.num_frames = math.floor(gop_size / 2)

        self.conv0 = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.convmerge = nn.Conv2d(self.num_frames * mid_channels, mid_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(mid_channels * 3, 3, 3, 1, 1)

        self.conv0_context = nn.Conv2d(context_channels, mid_channels, 3, 1, 1)
        self.convmerge_context = nn.Conv2d(self.num_frames * mid_channels, mid_channels, 1, 1, 0)

    def forward(self, x, context):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_1 = self.conv0(x)
        x_1 = rearrange(x_1, '(b t) c h w -> b t c h w', b=b)
        x_2 = self.convmerge(rearrange(x_1, 'b t c h w -> b (t c) h w'))
        x_2 = repeat(x_2.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.num_frames)

        context = rearrange(context, 'b c t h w -> (b t) c h w')
        x_1_context = self.conv0_context(context)
        x_1_context = rearrange(x_1_context, '(b t) c h w -> b t c h w', b=b)
        x_2_context = self.convmerge(rearrange(x_1_context, 'b t c h w -> b (t c) h w'))
        x_2_context = repeat(x_2_context.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.num_frames)

        x_4 = torch.cat([x_1, x_2, x_2_context], dim=2)
        x_4 = rearrange(x_4, 'b t c h w -> (b t) c h w')
        x_4 = self.conv2(x_4)
        out = x + x_4
        return rearrange(out, '(b t) c h w -> b c t h w', b=b)


class PFNLWithBlocks_old(nn.Module):
    def __init__(self, gop_size=7, input_channels=3, context_channels=12, mid_channels=12):
        super(PFNLWithBlocks_old, self).__init__()
        self.num_frames = math.floor(gop_size / 2)
        self.num_block = 6

        self.rfrb_blocks = nn.ModuleList([
            PFNL_old(gop_size=gop_size, input_channels=input_channels, 
                 context_channels=context_channels, mid_channels=mid_channels)
            for _ in range(self.num_block)
        ])

    def forward(self, x, context):
        for rfrb_block in self.rfrb_blocks:
            x = rfrb_block(x, context)
        return x

if __name__ == "__main__":
    pass
