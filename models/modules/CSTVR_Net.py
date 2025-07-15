

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
import os
from models.modules.Inv_arch import Quantization_H265_Stream,DenseBlockSurrogatedCodec,SurrogatedCodecInvINN,subnet,SurrogatedCodecIntraCNN,ycbcr444_to_420,ycbcr420_planner_to_444,decode_compressed_video,resize_flow,torch_warp,Quantization
from utils.functions import ycbcr2rgb, rgb2ycbcr
from torch.nn.parallel import DataParallel, DistributedDataParallel

class CSTVR_w_surrogate_net(nn.Module):
    def __init__(self, opt, subnet_constructor=None, down_num=2):
        super(CSTVR_w_surrogate_net, self).__init__()
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        from models.VFI_models.cvrs.arch.IMSM import IND_inv3D
        from models.VFI_models.cvrs.utils.options import yaml_load
        inv_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/config.yml')['network_g']['opt']
        self.model = IND_inv3D(inv_opt).to('cuda')
        inv_weight_p = os.path.join('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/inverter/model.pth')
        inv_weight = torch.load(inv_weight_p)
        self.model.load_state_dict(inv_weight['params'], strict=True)

        time_factor = 2
        scale_factor = 1
        rescale_opt = yaml_load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/config.yml')
        if time_factor == 2 and scale_factor == 1:
            from models.VFI_models.cvrs.arch.Mynet_arch import RescalerNet
        else:
            from models.VFI_models.cvrs.arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
        self.rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to('cuda')
        weight = torch.load('/data/fengxm/vimeo90k/pretrained_model/CVRS/Tx2_Sx1_vimeo/rescaler/model.pth')
        self.rescale_model.load_state_dict(weight['params'], strict=True)
        self.rescale_model.eval()

        self.opt = opt

        self.Quantization_H265_codec = Quantization_H265_Stream(None, -1, None, opt)
        self.quantization = Quantization()

        # compression simulation和restoration结合
        if opt['surr_type'] == 'Meta-assisted':
            subnet_type = "MetaLightDBNet"
        else:
            subnet_type = "LightDBNet"
        # inter coding
        if self.opt['network_G']['surrogated_model'] == 'DenseBlock':
            self.Quantization_H265_Suggrogate = DenseBlockSurrogatedCodec(opt, subnet(subnet_type, 'xavier', gc=8), down_num, surr_type=opt['surr_type'], surr_model=opt['surr_model'])
        elif self.opt['network_G']['surrogated_model'] == 'TINN':
            self.Quantization_H265_Suggrogate = SurrogatedCodecInvINN(opt, subnet(subnet_type, 'xavier', gc=8), down_num, surr_type=opt['surr_type'], surr_model=opt['surr_model'])
        else:
            raise Exception('invalid type.')
        # intra coding
        self.Intra_H265_Surrogate = SurrogatedCodecIntraCNN(opt)
        self.surr_mode = opt['surr_mode']
        self.load()
    
    def load(self):
        load_path_S = self.opt['path']['pretrain_model_S']
        if load_path_S is not None:
            self.load_network(load_path_S, self.Quantization_H265_Suggrogate, self.opt['path']['strict_load'])
        load_path_S = self.opt['path']['pretrain_model_S_Intra']
        if load_path_S is not None:
            self.load_network(load_path_S, self.Intra_H265_Surrogate, self.opt['path']['strict_load'])

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        if self.rank == -1:
            load_net = torch.load(load_path)
        else:
            load_net = torch.load(load_path,map_location='cuda:{}'.format(self.rank))
            print("loading model from cuda:", self.rank)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            # if not k.startswith('irn.'):
            # 	continue
            if k.startswith('module.'):
                load_net_clean[ k[7:]] = v
            else:
                load_net_clean[  k] = v
        network.load_state_dict(load_net_clean, strict=strict)
  

    def noise_video_surrogate(self, LR, qp=None, pix_fmt=None, inplace_flag=False, init_I_frame=None, test_mode=False, only_codec=False):
        if self.opt['codec'] == 'h265' or test_mode:
            if self.opt['network_G']['surrogated_model'] == 'DenseBlock':
                # b c t h w
                LR = rearrange(LR, 'b c t h w -> b t c h w')
                b, t, c, h, w = LR.shape
                with torch.no_grad():
                    if init_I_frame is None:
                        LR_input = LR.detach().cpu().numpy()
                    else:
                        LR_input = torch.cat([ref_lq.unsqueeze(1), LR], dim=1).detach().cpu().numpy()
                    input_array = []
                    LR_input = rearrange(LR_input, 'b t c h w -> (b t) c h w')
                    # assert LR.shape[0] == 1
                    for frm_id in range(LR_input.shape[0]):
                        y, uv = ycbcr444_to_420(LR_input.astype(np.float)[frm_id])
                        yuv420_image = np.concatenate([y, uv.reshape(1, h // 2, w)], axis=1)[0]
                        input_array.append((yuv420_image * 255).astype(np.uint8))
                    input_array = np.asarray(input_array)

                    h265_path = self.Quantization_H265_codec.open_writer('cpu', w, h, pix_fmt='yuv420p', verbosity=0, mode='cqp', qp=qp[0].item())
                    self.Quantization_H265_codec.write_multi_frames(input_array)

                    _, img_distri = self.Quantization_H265_codec.close_writer()

                    self.Quantization_H265_codec.open_reader(verbosity=0, pix_fmt='yuv420p')
                    out_array = self.Quantization_H265_codec.read_multi_frames(input_array.shape[0])

                    YCbCr444_compressed = []
                    for frame in out_array:
                        YCbCr444_compressed.append(ycbcr420_planner_to_444(frame, h, w))
                    compress_label = torch.tensor(YCbCr444_compressed).to(LR.device) / 255.
                    compress_label = rearrange(compress_label, 'b h w c -> b c h w').contiguous().detach()
                high_frequency_list = torch.tensor(1.0).to(LR.device)
                self.compress_label = rearrange(compress_label, '(b t) c h w -> b t c h w', b=b)
                self.LR = LR


                LR = rearrange(LR, 'b t c h w -> (b t) c h w')
                LR_surrogated = self.Quantization_H265_Suggrogate(x=LR)


            elif self.opt['network_G']['surrogated_model'] == 'TINN':
                LR = rearrange(LR, 'b c t h w -> b t c h w')
                b, t, c, h, w = LR.shape
                seq_num, frm_num = LR.shape[0], LR.shape[1]
                LR_surrogated = torch.zeros_like(LR)
                ref_lq = None if init_I_frame is None else init_I_frame[:, 0]
                with torch.no_grad():
                    if init_I_frame is None:
                        LR_input = LR.detach().cpu().numpy()
                    else:
                        LR_input = torch.cat([ref_lq.unsqueeze(1), LR], dim=1).detach().cpu().numpy()
                    input_array = []
                    LR_input = rearrange(LR_input, 'b t c h w -> (b t) c h w')
                    # assert LR.shape[0] == 1
                    for frm_id in range(LR_input.shape[0]):
                        y, uv = ycbcr444_to_420(LR_input.astype(np.float)[frm_id])
                        yuv420_image = np.concatenate([y, uv.reshape(1, h // 2, w)], axis=1)[0]
                        input_array.append((yuv420_image * 255).astype(np.uint8))
                    input_array = np.asarray(input_array)

                    h265_path = self.Quantization_H265_codec.open_writer('cpu', w, h, pix_fmt='yuv420p', verbosity=0, mode='cqp', qp=qp[0].item())
                    self.Quantization_H265_codec.write_multi_frames(input_array)

                    _, img_distri = self.Quantization_H265_codec.close_writer()

                    self.Quantization_H265_codec.open_reader(verbosity=0, pix_fmt='yuv420p')
                    out_array = self.Quantization_H265_codec.read_multi_frames(input_array.shape[0])

                    YCbCr444_compressed = []
                    for frame in out_array:
                        YCbCr444_compressed.append(ycbcr420_planner_to_444(frame, h, w))
                    compress_label = torch.tensor(YCbCr444_compressed).to(LR.device) / 255.
                    compress_label = rearrange(compress_label, 'b h w c -> b c h w').contiguous().detach()
                    # ref_lq = compress_label[0:1]
                    if only_codec:
                        return compress_label[1:], img_distri

                cvf = decode_compressed_video(h265_path, format='yuv420p')
                residual = torch.tensor(cvf['residual'].astype(np.float32)).to(LR.device)  # (7, 256, 448, 3)
                mv_x_L0 = torch.tensor(cvf['mv_x_L0'].astype(np.float32)).to(LR.device)  # (7, 64, 112)
                mv_y_L0 = torch.tensor(cvf['mv_y_L0'].astype(np.float32)).to(LR.device)

                mv = torch.stack([mv_x_L0, mv_y_L0], dim=1)  # t c h w
                res = rearrange(residual, 'b h w c -> b c h w').unsqueeze(2) / 255.
                mv = resize_flow(mv, size_type='ratio', sizes=(4, 4)) / 16.
                mv = rearrange(mv, '(b t) c h w -> b t c h w', b=b)
                # save state for optimizer of surrogate loss
                self.compress_label = rearrange(compress_label, '(b t) c h w -> b t c h w', b=b)
                self.LR = LR
                self.qp = qp.to(LR.device)
                self.mv = mv
                self.res = res

                # mimick_loss = torch.tensor(0).to(LR.device)
                high_frequency_list = []
                if not test_mode:
                    for seq_id in range(seq_num):
                        if init_I_frame is None:
                            ref_lq = self.Intra_H265_Surrogate(LR[seq_id,0:1], qp.to(LR.device))
                            LR_surrogated[seq_id, 0:1] = ref_lq
                            # mimick_loss = mimick_loss + F.l1_loss(ref_lq, self.compress_label[:,0])
                        else:
                            ref_lq = self.compress_label[seq_id, 0:1]
                        for i in range(1, frm_num):
                            current_hq, current_mv, lq_label = LR[seq_id,i], mv[seq_id, i:i+1], self.compress_label[seq_id, i:i+1]
                            high_frequency = current_hq - torch_warp(ref_lq, current_mv)
                            high_frequency_list.append(high_frequency) 
                            low_frequency = ref_lq + torch_warp(high_frequency, -current_mv)
                            # forward process
                            compress_output = self.Quantization_H265_Suggrogate(x=torch.cat([low_frequency, high_frequency], dim=1), model='compression', qp=qp[0:1])
                            # backward process
                            lq_recon = self.Quantization_H265_Suggrogate(x=compress_output, rev=True, model='compression', qp=qp[0:1])
                            low_frequency_recon, high_frequency_recon = lq_recon[:, :3], lq_recon[:, 3:]
                            low_frequency_recon = low_frequency_recon - torch_warp(high_frequency_recon, -current_mv)
                            high_frequency_recon = high_frequency_recon + torch_warp(low_frequency_recon, current_mv)
                            high_frequency_recon = high_frequency_recon.clamp(0, 1)

                            ref_lq = high_frequency_recon
                            LR_surrogated[seq_id, i] = high_frequency_recon
                    high_frequency_list = torch.stack(high_frequency_list, dim=1)  # (b,t,c,h,w)
                LR_surrogated = rearrange(LR_surrogated, 'b t c h w -> (b t) c h w')
            else:
                raise Exception('invalid type.')
                
            # x265 encoding
            bpp = torch.tensor([img_distri])
            bpp = bpp.to(LR.device)
            if inplace_flag or test_mode:
                # 类似于SelfC的H265_xxx
                if init_I_frame is None:
                    LR_compressed = rearrange(self.compress_label, 'b t c h w -> (b t) c h w').contiguous()
                else:
                    LR_compressed = rearrange(self.compress_label[:,1:], 'b t c h w -> (b t) c h w').contiguous()
                LR_surrogated.data.copy_(LR_compressed) 
            else:
                LR_compressed = LR_surrogated
            LR_surrogated = rearrange(LR_surrogated, '(b t) c h w -> b t c h w', b=b)
            LR_compressed = rearrange(LR_compressed, '(b t) c h w -> b t c h w', b=b)
        elif self.opt['codec'] == 'none':
            # ablation: remove surrogate network
            LR_compressed = rearrange(self.quantization(LR)[0], 'c t h w -> t c h w')
            LR_surrogated = rearrange(self.quantization(LR)[0], 'c t h w -> t c h w')
            bpp = torch.tensor(1.0).to(LR.device)
            high_frequency_list = torch.tensor(1.0).to(LR.device)
        else:
            raise Exception('invalid type.')

        return LR_surrogated, bpp




    def calculate_mimick_loss(self):
        # self.compress_label, self.LR
        # 断开计算图连接，但是也可以继续反向梯度传播
        self.compress_label = self.compress_label.detach().clone().requires_grad_(True)
        self.LR = self.LR.detach().clone().requires_grad_(True)
        mimick_loss = torch.tensor(0).to(self.LR.device)

        if self.opt['network_G']['surrogated_model'] == 'DenseBlock':
            # b c t h w
            # LR = rearrange(LR, 'b c t h w -> (b t) c h w')
            LR_surrogated = self.Quantization_H265_Suggrogate(x=rearrange(self.LR, 'b t c h w -> (b t) c h w'))
            mimick_loss = F.l1_loss(LR_surrogated, rearrange(self.compress_label, 'b t c h w -> (b t) c h w'))

        elif self.opt['network_G']['surrogated_model'] == 'TINN':
            self.qp = self.qp.detach().clone().requires_grad_(True)
            self.mv = self.mv.detach().clone().requires_grad_(True)
            high_frequency_list = []
            seq_id = 0
            ref_lq = self.Intra_H265_Surrogate(self.LR[seq_id,0:1], self.qp)
            mimick_loss = mimick_loss + F.l1_loss(ref_lq, self.compress_label[:,0])
            for i in range(1, self.LR.shape[1]):
                current_hq, current_mv, lq_label = self.LR[seq_id,i], self.mv[seq_id, i:i+1], self.compress_label[seq_id, i:i+1]
                high_frequency = current_hq - torch_warp(ref_lq, current_mv)
                high_frequency_list.append(high_frequency) 
                low_frequency = ref_lq + torch_warp(high_frequency, -current_mv)
                # forward process
                compress_output = self.Quantization_H265_Suggrogate(x=torch.cat([low_frequency, high_frequency], dim=1), model='compression', qp=self.qp[0:1])
                out_lrs = compress_output[:, :3, :, :]
                l_forw_fit_compress = F.l1_loss(out_lrs, low_frequency) * 10  # LR guidance
                # backward process
                lq_recon = self.Quantization_H265_Suggrogate(x=compress_output, rev=True, model='compression', qp=self.qp[0:1])
                low_frequency_recon, high_frequency_recon = lq_recon[:, :3], lq_recon[:, 3:]
                low_frequency_recon = low_frequency_recon - torch_warp(high_frequency_recon, -current_mv)
                high_frequency_recon = high_frequency_recon + torch_warp(low_frequency_recon, current_mv)
                high_frequency_recon = high_frequency_recon.clamp(0, 1)
                l_back_rec_compress = F.l1_loss(high_frequency_recon, lq_label) * 40
                # loss
                weight = ((current_hq - lq_label).abs() > 1e-2) * 10.
                l_back_rec_compress_weight = F.l1_loss(high_frequency_recon * weight, lq_label * weight) * 40
                mimick_loss = mimick_loss + l_forw_fit_compress + l_back_rec_compress_weight + l_back_rec_compress
                ref_lq = high_frequency_recon
        return mimick_loss
    

    
    def forward(self, x, rev=False, qp=None, pix_fmt=None, init_I_frame=None,inplace_flag=False, train_surrogate=False, close_codec=False):
        if train_surrogate:
            return self.calculate_mimick_loss()

        # x.shape b,t,c,h,w
        if self.training:
            if not rev:
                b, t, c, h, w = x.shape
                down_size = (4, x.shape[-2] , x.shape[-1])
                x_down = self.rescale_model(rearrange(ycbcr2rgb(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b), down_size, rev=False)
                # x_down = self.rescale_model.inference_down(rearrange(ycbcr2rgb(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b), down_size)
                LR_img = self.model(x_down, latent_to_RGB=True)
                LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
                LF_surrogated, bpp = self.noise_video_surrogate(rearrange(LF, 'b t c h w -> b c t h w').clamp(0,1), pix_fmt=pix_fmt, qp=qp, init_I_frame=init_I_frame,inplace_flag=inplace_flag)  # (b,c,t,h,w)
                return LF, LF_surrogated, bpp
            else:
                y, _ = x
                b = y.shape[0]
                rev_back = self.model(rearrange(ycbcr2rgb(rearrange(y, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b), latent_to_RGB=False)
                out_x = self.rescale_model(rev_back, (self.opt['gop'], y.shape[-2], y.shape[-1]), rev=True)
                out_x = rearrange(out_x, 'b c t h w -> b t c h w')
                return out_x
        else:
            if not rev:
                b, t, c, h, w = x.shape
                down_size = (4, x.shape[-2] , x.shape[-1])
                x_down = self.rescale_model(rearrange(ycbcr2rgb(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b), down_size, rev=False)
                LR_img = self.model(x_down, latent_to_RGB=True)
                LF = rearrange(rgb2ycbcr(rearrange(LR_img, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
                return LF
            else:
                y, _ = x
                b = y.shape[0]
                rev_back = self.model(rearrange(ycbcr2rgb(rearrange(y, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b), latent_to_RGB=False)
                out_x = self.rescale_model(rev_back, (self.opt['gop'], y.shape[-2], y.shape[-1]), rev=True)
                out_x = rearrange(out_x, 'b c t h w -> b t c h w')
                return out_x
            
