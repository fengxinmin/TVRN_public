import os
import time
import numpy as np
import torch
import torch.nn as nn
import skvideo.io
from global_var import GlobalVar


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image with shape (*, 3, H, W).

    Returns:
        torch.Tensor: YCbCr version of the image.
    """
    if not torch.is_tensor(image):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input must have shape (*, 3, H, W). Got {image.shape}")

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack((y, cb, cr), dim=-3)


class QuantizationH265Stream:
    def __init__(self, q=17, keyint=12, scale_times=2, opt=None):
        self.q = q
        self.keyint = -1  # force no keyint unless set
        self.scale_times = scale_times
        QuantizationH265Stream.file_random_name = None
        self.h265_all_default = opt["network_G"]["h265_all_default"]
        self.video_frame_num = 0

    def open_writer(self, dev_id, w, h, pix_fmt, verbosity=1, no_info=1, extra_info=None, mode='cqp'):
        assert mode in ['crf', 'cqp']
        if not QuantizationH265Stream.file_random_name:
            QuantizationH265Stream.file_random_name = str(time.time())

        self.w, self.h, self.pix_fmt = w, h, pix_fmt
        self.video_frame_num = 0

        suffix = f"{dev_id}{extra_info if extra_info else ''}cvb.h265"
        self.video_name = os.path.join("/output", f"outputvideo{suffix}")

        if os.path.exists(self.video_name):
            os.remove(self.video_name)

        # x265 encoding params
        if self.keyint and self.keyint > 0:
            x265_param = f"{mode}={self.q}:keyint={self.keyint}:no-info={no_info}"
        else:
            x265_param = f"{mode}={self.q}:no-info={no_info}"

        output_dict = {
            "-s": f"{w}x{h}",
            "-pix_fmt": pix_fmt,
            "-c:v": "libx265",
            "-x265-params": x265_param
        }

        if not self.h265_all_default:
            output_dict.update({
                "-preset": "veryfast",
                "-tune": "zerolatency"
            })

        input_dict = {"-s": f"{w}x{h}", "-pix_fmt": pix_fmt} if pix_fmt == 'rgb24' else {}
        if pix_fmt not in ['rgb24', 'yuv444p']:
            raise NotImplementedError("Unsupported pixel format.")

        self.rgb_yuv_writer = skvideo.io.FFmpegWriter(
            self.video_name, inputdict=input_dict, outputdict=output_dict, verbosity=verbosity
        )

    def write_multi_frames(self, input_tensor):
        input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
        output_uint8 = (input_tensor * 255.0).round().byte()
        b, c, h, w = output_uint8.shape
        self.h, self.w = h, w

        frames = output_uint8.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(c // 3):
            frame = frames[0, :, :, i * 3:(i + 1) * 3]
            self.rgb_yuv_writer.writeFrame(frame)
            self.video_frame_num += 1

    def close_writer(self):
        self.rgb_yuv_writer.close()
        file_size = os.path.getsize(self.video_name)
        bpp = file_size * 8.0 / (self.h * self.w * self.video_frame_num)
        self.video_frame_num = 0
        return torch.tensor([bpp]), bpp

    def open_reader(self, verbosity=1):
        self.reader = skvideo.io.FFmpegReader(
            self.video_name, inputdict={}, outputdict={}, verbosity=verbosity
        )

    def read_multi_frames(self, num):
        decoded_frames = []
        for count, frame in enumerate(self.reader.nextFrame()):
            frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
            decoded_frames.append(frame_tensor)
            if count + 1 == num:
                break

        stacked = torch.stack(decoded_frames, dim=0)
        return stacked.permute(0, 3, 1, 2)
