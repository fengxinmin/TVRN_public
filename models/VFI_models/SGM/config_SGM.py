from functools import partial
import torch.nn as nn

from SGM import feature_extractor
from SGM import flow_estimation

'''==========Model config=========='''


def init_model_config(F=32, W=7, depth=[2, 2, 2, 4], num_key_points=0.):
    '''This function should not be modified'''
    return {
        'embed_dims': [F, 2 * F, 4 * F, 8 * F],
        'num_heads': [8 * F // 32],
        'mlp_ratios': [4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W]
    }, {
        'embed_dims': [F, 2 * F, 4 * F, 8 * F],
        'motion_dims': [0, 0, 0, 8 * F // depth[-1]],
        'depths': depth,
        'scales': [8],
        'hidden_dims': [4 * F],
        'c': F,
        'num_key_points': num_key_points,
    }

MODEL_CONFIG = {
    'LOGNAME': 'ours-small-1-2',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        F=16,
        W=7,
        depth=[2, 2, 2, 4],
        num_key_points=0.5
    )
}
# MODEL_CONFIG = {
#     'LOGNAME': 'ours-base-1-2',
#     'MODEL_TYPE': (feature_extractor, flow_estimation),
#     'MODEL_ARCH': init_model_config(
#         F=32,
#         W=7,
#         depth=[2, 2, 2, 6],
#         num_key_points=0.5
#     )
# }