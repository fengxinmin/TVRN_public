import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
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
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class DenseBlock3D(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5
    
class LightDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(LightDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        # self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 3 * gc, channel_out, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3), 1))

        return x5


class ExtremeLightDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(ExtremeLightDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, ], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2,], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2), 1))

        return x5


class MetaLightDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(MetaLightDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias, padding_mode='replicate')
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias, padding_mode='replicate')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.qp_module_1 = nn.Linear(1, channel_in + 2 * gc)
        self.qp_module_2 = nn.Linear(1, channel_in + 2 * gc)
        # self.inst_norm5 = nn.InstanceNorm2d(channel_out, affine=True)
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, ], 0.1)
            mutil.initialize_weights_xavier([self.qp_module_1, self.qp_module_2], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, ], 0.1)
            mutil.initialize_weights([self.qp_module_1, self.qp_module_2], 0.1)
        mutil.initialize_weights(self.conv5, 0)
        # mutil.initialize_weights(self.inst_norm5, 0.1)

    def forward(self, x, qp):
        # qp.shape (N,1)
        qp_plus = self.qp_module_1(qp).unsqueeze(-1).unsqueeze(-1)
        qp_mul = self.qp_module_2(qp).unsqueeze(-1).unsqueeze(-1)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x_total = torch.cat((x, x1, x2), 1)
        x_total += qp_plus
        x_total *= qp_mul
        x5 = self.conv5(x_total)
        # x5 = self.inst_norm5(self.conv5(x_total))
        return x5
    

def subnet(net_structure, init='xavier', gc=16):
    def constructor(channel_in, channel_out, gc=gc):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return DenseBlock(channel_in, channel_out, gc=gc)
        elif net_structure == 'LightDBNet':
            if init == 'xavier':
                return LightDenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return LightDenseBlock(channel_in, channel_out, gc=gc)
            
        elif net_structure == 'ExtremeLightDBNet':
            if init == 'xavier':
                return ExtremeLightDenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return ExtremeLightDenseBlock(channel_in, channel_out, gc=gc)

        elif net_structure == 'MetaLightDBNet':
            if init == 'xavier':
                return MetaLightDenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return MetaLightDenseBlock(channel_in, channel_out, gc=gc)
            
        if net_structure == 'DBNet3D':
            if init == 'xavier':
                return DenseBlock3D(channel_in, channel_out, init, gc=gc)
            else:
                return DenseBlock3D(channel_in, channel_out, gc=gc)

        else:
            return None

    return constructor
