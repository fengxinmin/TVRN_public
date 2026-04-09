from models_cvrs.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
from models_cvrs.arch.Mynet_arch import STREV_down, STREV_up
from models_cvrs.arch.InvNN_3D_arch import my_InvNN
from models_cvrs.module.inv_module import D2DTInput_dense
from models_cvrs.module.Quantization import Quantize_ste
from models_cvrs.module.general_module import SpaceTimePixelShuffle,SpaceTimePixelUnShuffle


@ARCH_REGISTRY.register()
class IND_inv3D(nn.Module):
    def __init__(self,opt):
        super(IND_inv3D, self).__init__()
        self.opt = opt
        self.downsample = STREV_down(opt['down_opt'])
        # self.upsample = STREV_up(opt['up_opt'])
        self.quant_type = opt['down_opt']['quant_type']
        self.inv_block =  my_InvNN(mode = 'train',channel_in=3,subnet_constructor =D2DTInput_dense ,block_num=opt['block_num'])
        self.quan_layer = Quantize_ste(min_val=0.0,max_val=1.0)
    
        self.st_shuffle = SpaceTimePixelShuffle(r=1,s=2)
        self.st_unshuffle = SpaceTimePixelUnShuffle(r=1,s=2)
    @torch.no_grad()
    def inference_down(self,imgs,down_size):
        x_down = self.downsample(imgs,down_size)
        return x_down
    
    # @torch.no_grad()
    # def inference_up(self,imgs,up_size):
    #     up_imgs = self.upsample(imgs,target_size=up_size)
    #     up_imgs = torch.clamp(up_imgs, 0, 1)
    #     return up_imgs
    
    @torch.no_grad()
    def inference_latent2RGB(self,x_down):
        LR_img_stack,jac = self.inv_block(x_down,cal_jacobian = True)
        LR_img = self.st_shuffle(self.st_shuffle(LR_img_stack))
        LR_img = torch.clamp(LR_img, 0, 1)
        return LR_img
    @torch.no_grad()
    def inference_RGB2latent(self,LR_img):
        LR_latent = self.st_unshuffle(self.st_unshuffle(LR_img))
        rev_back = self.inv_block(LR_latent,rev = True)
        rev_back = torch.clamp(rev_back, 0, 1)
        return rev_back

    def forward(self,x,latent_to_RGB=False):
        if latent_to_RGB:
            LR_img_stack,jac = self.inv_block(x,cal_jacobian = True)
            LR_img = self.st_shuffle(self.st_shuffle(LR_img_stack))
            LR_img = torch.clamp(LR_img, 0, 1)
            return LR_img
        else:
            LR_latent = self.st_unshuffle(self.st_unshuffle(x))
            rev_back = self.inv_block(LR_latent,rev = True)
            rev_back = torch.clamp(rev_back, 0, 1)
            return rev_back
