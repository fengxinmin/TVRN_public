'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os.path as osp
import pickle
import random
import os
import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

try:
	import mc  # import memcached
except ImportError:
	pass
logger = logging.getLogger('base')


class Vimeo90KDataset(data.Dataset):
	'''
	Reading the training Vimeo90K dataset
	key example: 00001_0001 (_1, ..., _7)
	GT (Ground-Truth): 4th frame;
	LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
	'''

	def __init__(self, opt):
		super(Vimeo90KDataset, self).__init__()
		self.opt = opt
		# temporal augmentation
		self.interval_list = opt['interval_list']
		self.random_reverse = opt['random_reverse']
		logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
			','.join(str(x) for x in opt['interval_list']), self.random_reverse))

		self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
		self.data_type = self.opt['data_type']
		# self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
		self.LR_input = False

		#### determine the LQ frame list
		'''
		N | frames
		1 | 4
		3 | 3,4,5
		5 | 2,3,4,5,6
		7 | 1,2,3,4,5,6,7
		'''
		self.LQ_frames_list = []
		for i in range(opt['N_frames']):
			self.LQ_frames_list.append(i + (9 - opt['N_frames']) // 2)

		#### directly load image keys
		if self.data_type == 'lmdb':
			self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
			logger.info('Using lmdb meta info for cache keys.')
		elif opt['cache_keys']:
			logger.info('Using cache keys: {}'.format(opt['cache_keys']))
			self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
		else:
			raise ValueError(
				'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')
		assert self.paths_GT, 'Error: GT path is empty.'

		if self.data_type == 'lmdb':
			self.GT_env, self.LQ_env = None, None
		elif self.data_type == 'mc':  # memcached
			self.mclient = None
		elif self.data_type == 'img':
			pass
		else:
			raise ValueError('Wrong data type: {}'.format(self.data_type))

	def _init_lmdb(self):
		# https://github.com/chainer/chainermn/issues/129
		self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
								meminit=False)
		self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
								meminit=False)
		# self.LQ_env = lmdb.open(os.path.join(self.opt['dataroot_LQ'], 'vimeo90k_train_Q' + str(self.opt['qp']) + '.lmdb'), readonly=True, lock=False, readahead=False, meminit=False)

	def _ensure_memcached(self):
		if self.mclient is None:
			# specify the config files
			server_list_config_file = None
			client_config_file = None
			self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
														  client_config_file)

	def _read_img_mc(self, path):
		''' Return BGR, HWC, [0, 255], uint8'''
		value = mc.pyvector()
		self.mclient.Get(path, value)
		value_buf = mc.ConvertBuffer(value)
		img_array = np.frombuffer(value_buf, np.uint8)
		img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
		return img

	def __getitem__(self, index):
		if self.data_type == 'mc':
			self._ensure_memcached()
		elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
			self._init_lmdb()

		key = self.paths_GT[index]
		name_a, name_b = key.split('_')
		img_GT_l = []
		img_LQ_l = []
		for v in self.LQ_frames_list:
			img_GT = util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448))
			img_LQ = util.read_img(self.LQ_env, name_a + '_{}'.format(v), (3, 256, 448))
			img_LQ_l.append(img_LQ)
			img_GT_l.append(img_GT)

		# if self.opt['phase'] == 'train':
		# 	GT_size = 256
		# 	H, W = img_LQ.shape[0], img_LQ.shape[1]
		# 	img_LQ_l += img_GT_l
		# 	# random crop
		# 	rnd_h = random.randint(0, max(0, H - GT_size))
		# 	rnd_w = random.randint(0, max(0, W - GT_size))
		# 	rlt = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
		# 	# augmentation - flip, rotate
		# 	rlt = util.augment(rlt, self.opt['use_flip'], self.opt['use_rot'])
		# 	rlt_len = len(rlt)
		# 	img_LQ_l = rlt[0:rlt_len//2]
		# 	img_GT_l = rlt[rlt_len // 2:]


		# stack LQ images to NHWC, N is the frame number
		# img_LQs = np.stack(img_LQ_l, axis=0)
		img_GTs = np.stack(img_GT_l, axis=0)
		img_LQs = np.stack(img_LQ_l, axis=0)
		# BGR to RGB, HWC to CHW, numpy to tensor
		img_GTs = img_GTs[:, :, :, [2, 1, 0]]
		# img_LQs = img_LQs[:, :, :, [2, 1, 0]]  # LQ是YUV排列的
		img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
		img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()
		# return {'LQ': img_LQs, 'GT': img_GTs, 'GT_path': key}
		return {'GT': img_GTs, 'LQ': img_LQs}

	def __len__(self):
		return len(self.paths_GT)
