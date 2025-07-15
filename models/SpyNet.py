import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature


def torch_warp(tensorInput, tensorFlow):
    # backward warp
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput, grid=grid.permute(0, 2, 3, 1), mode='bilinear',
                                               padding_mode='border')  # return torch.nn.functional.grid_sample(input=tensorInput,  #                                        grid=grid.permute(0, 2, 3, 1),  #                                        mode='bilinear',  #                                        padding_mode='border',  #                                        align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput, grid=grid.permute(0, 2, 3, 1), mode='bilinear',
                                               padding_mode='border')  # return torch.nn.functional.grid_sample(input=tensorInput,  #                                        grid=grid.permute(0, 2, 3, 1),  #                                        mode='bilinear',  #                                        padding_mode='border',  #                                        align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def load_weight_form_np(me_model_dir, layername):
    index = layername.find('modelL')
    if index == -1:
        print('load models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = me_model_dir + name + '-weight.npy'
        modelbias = me_model_dir + name + '-bias.npy'
        weightnp = np.load(modelweight)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)


class MEBasic(nn.Module):
    def __init__(self, me_model_dir, layername):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

        if me_model_dir is None:
            return
        self.conv1.weight.data, self.conv1.bias.data = load_weight_form_np(me_model_dir, layername + '_F-1')
        self.conv2.weight.data, self.conv2.bias.data = load_weight_form_np(me_model_dir, layername + '_F-2')
        self.conv3.weight.data, self.conv3.bias.data = load_weight_form_np(me_model_dir, layername + '_F-3')
        self.conv4.weight.data, self.conv4.bias.data = load_weight_form_np(me_model_dir, layername + '_F-4')
        self.conv5.weight.data, self.conv5.bias.data = load_weight_form_np(me_model_dir, layername + '_F-5')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self, me_model_dir, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic(me_model_dir, layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])  # print(self.moduleBasic)

    def forward(self, im1, im2, Lm=0, init_flow=None):
        # img1: 当前帧， img2: 参考帧, Lm: 输入帧所在的下采样倍率（0，1，2，3）， 0 means 没有下采样
        batchsize = im1.size()[0]

        im1list = []
        im2list = []
        for intLevel in range(self.L):
            if intLevel < Lm:
                # 该深度不会被使用，例如qt-net不会使用0,1, mtt-mask不会使用0
                im1list.append(None)
                im2list.append(None)
            else:
                im1list.append(F.avg_pool2d(im1, kernel_size=2 ** (intLevel - Lm), stride=2 ** (intLevel - Lm)))
                im2list.append(F.avg_pool2d(im2, kernel_size=2 ** (intLevel - Lm), stride=2 ** (intLevel - Lm)))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device = im1.device
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device)
        flow_list = []
        for intLevel in range(self.L):
            if intLevel > 3 - Lm:
                flow_list.append(None)
            else:
                flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
                flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel], flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample), flowfiledsUpsample], 1))
                flow_list.append(flowfileds)

        return flow_list
