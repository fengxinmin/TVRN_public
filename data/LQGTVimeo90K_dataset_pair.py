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
from einops import rearrange
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
		self.LQ_env = {}
		for qp in self.opt['qp_list']:
			self.LQ_env[str(qp)] = lmdb.open(self.opt['dataroot_GT'].replace('GT', f'Q{qp}'), readonly=True, lock=False, readahead=False, meminit=False)
		self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
								meminit=False)
		# self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
		# 						meminit=False)
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
		if self.opt['plt_tsne']:
			# 输出某个index的全部P帧, 这里的index设置为42
			results_dict = {}
			# key = self.paths_GT[index]
			key = '42_1'
			name_a, name_b = key.split('_')
			img_GT_l = []
			LQ_frames_list = [index+1, ]

			for v in LQ_frames_list:
				img_GT_l.append(util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448)))
			img_GTs = np.stack(img_GT_l, axis=0)
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array
			results_dict['GT'] = img_GTs
			name_list = ['a', 'b', 'c', 'd', 'e']
			for qp_id, qp in enumerate([17,22,27,32,37]):
				img_LQ_l_a = []
				for v in LQ_frames_list:
					img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQs_a = np.stack(img_LQ_l_a, axis=0)
				img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()
				results_dict[f'LQ_{name_list[qp_id]}'] = img_LQs_a
				results_dict[f'qp_{name_list[qp_id]}'] = qp
			return results_dict
		if self.opt['out_total_qp']:
			# 输出五种QP对应的划分损失
			results_dict = {}
			key = self.paths_GT[index]
			name_a, name_b = key.split('_')
			img_GT_l = []
			if self.opt['pretrain_ranker']:
				LQ_frames_list = [random.randint(1, 6),]
			else:
				LQ_frames_list = self.LQ_frames_list

			for v in LQ_frames_list:
				img_GT_l.append(util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448)))
			img_GTs = np.stack(img_GT_l, axis=0)
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array
			results_dict['GT'] = img_GTs
			name_list = ['a', 'b', 'c', 'd', 'e']
			for qp_id, qp in enumerate([17,22,27,32,37]):
				img_LQ_l_a = []
				for v in LQ_frames_list:
					img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQs_a = np.stack(img_LQ_l_a, axis=0)
				img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()
				results_dict[f'LQ_{name_list[qp_id]}'] = img_LQs_a
				results_dict[f'qp_{name_list[qp_id]}'] = qp
			return results_dict
		elif self.opt['out_triplet_qp']:
			qp_1, qp_2, qp_3 = random.sample(self.opt['qp_list'], 3)

			key = self.paths_GT[index]
			name_a, name_b = key.split('_')
			img_GT_l = []
			img_LQ_l_a = []
			img_LQ_l_b = []
			img_LQ_l_c = []

			if self.opt['pretrain_ranker']:
				LQ_frames_list = [random.randint(1, 6),]
			else:
				LQ_frames_list = self.LQ_frames_list

			for v in LQ_frames_list:
				img_GT = util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448))
				img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp_1)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQ_l_b.append(util.read_img(self.LQ_env[str(qp_2)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQ_l_c.append(util.read_img(self.LQ_env[str(qp_2)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_GT_l.append(img_GT)

			img_GTs = np.stack(img_GT_l, axis=0)
			img_LQs_a = np.stack(img_LQ_l_a, axis=0)
			img_LQs_b = np.stack(img_LQ_l_b, axis=0)
			img_LQs_c = np.stack(img_LQ_l_c, axis=0)
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()
			img_LQs_b = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_b, (0, 3, 1, 2)))).float()
			img_LQs_c = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_c, (0, 3, 1, 2)))).float()

			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array
					
			return {'GT': img_GTs, 'LQ_a': img_LQs_a, 'LQ_b': img_LQs_b, 'qp_a': qp_1, 'qp_b': qp_2, 'LQ_c': img_LQs_c, 'qp_c': qp_3, }
		
		elif self.opt['out_aux_case']:
			# 给出另一张图片的qp_a压缩结果
			if len(self.opt['qp_list']) == 2:
				qp_1, qp_2 = self.opt['qp_list']
			else:
				# # 随机选择两个qp
				# qp_1, qp_2 = random.sample(self.opt['qp_list'], 2)
    
				# 随机选择一个qp, 另外一个qp也是随机的
				qp_1_id = random.randint(0,4)
				qp_1 = self.opt['qp_list'][qp_1_id]
				if qp_1 == 17:
					offset = 1
				elif qp_1 == 37:
					offset = -1
				else:
					offset = random.randint(0,1)
				qp_2 = self.opt['qp_list'][qp_1_id + offset]
	
			key = self.paths_GT[index]
			name_a, name_b = key.split('_')
			img_GT_l = []
			img_LQ_l_a = []
			img_LQ_l_b = []
			
			if self.opt['pretrain_ranker']:
				LQ_frames_list = [random.randint(1, 6),]
			else:
				LQ_frames_list = self.LQ_frames_list

			for v in LQ_frames_list:
				img_GT = util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448))
				img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp_1)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQ_l_b.append(util.read_img(self.LQ_env[str(qp_2)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_GT_l.append(img_GT)

			img_GTs = np.stack(img_GT_l, axis=0)
			img_LQs_a = np.stack(img_LQ_l_a, axis=0)
			img_LQs_b = np.stack(img_LQ_l_b, axis=0)
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()
			img_LQs_b = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_b, (0, 3, 1, 2)))).float()

			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array
			# aux_case
			index_aux = random.randint(0, len(self.paths_GT) - 1)
			key_aux = self.paths_GT[index_aux]
			name_a_aux, name_b = key_aux.split('_')
			img_GT_l_aux = []
			img_LQ_l_a_aux = []
			
			if self.opt['pretrain_ranker']:
				LQ_frames_list = [random.randint(1, 6),]
			else:
				LQ_frames_list = self.LQ_frames_list

			for v in LQ_frames_list:
				img_GT_aux = util.read_img(self.GT_env, name_a_aux + '_{}'.format(v), (3, 256, 448))
				img_LQ_l_a_aux.append(util.read_img(self.LQ_env[str(qp_1)], name_a_aux + '_{}'.format(v), (3, 256, 448)))
				img_GT_l_aux.append(img_GT_aux)

			img_GTs_aux = np.stack(img_GT_l_aux, axis=0)
			img_LQs_a_aux = np.stack(img_LQ_l_a_aux, axis=0)
			img_GTs_aux = img_GTs_aux[:, :, :, [2, 1, 0]]
			img_GTs_aux = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs_aux, (0, 3, 1, 2)))).float()
			img_LQs_a_aux = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a_aux, (0, 3, 1, 2)))).float()

			if self.opt['rgb_2_yuv444p']:
				output_array_aux = np.zeros_like(img_GTs_aux)
				for d in range( img_GTs_aux.shape[0]):
					rgb_image_aux =  rearrange(img_GTs_aux[d], 'c h w -> h w c')
					yuv_image_aux = cv2.cvtColor(rgb_image_aux.numpy(), cv2.COLOR_RGB2YUV)
					output_array_aux[d] = rearrange(yuv_image_aux, 'h w c -> c h w')
				img_GTs_aux = output_array_aux
    
			return {'GT': img_GTs, 'LQ_a': img_LQs_a, 'LQ_b': img_LQs_b, 'qp_a': qp_1, 'qp_b': qp_2, 'GT_aux': img_GTs_aux, 'LQ_a_aux': img_LQs_a_aux,}
		
		elif self.opt['out_single_seq']:
			qp_1_list = random.sample(self.opt['qp_list'], 1)
			qp_1 = qp_1_list[0]
	
			key = self.paths_GT[index]
			name_a, name_b = key.split('_')
			img_GT_l = []
			img_LQ_l_a = []
			img_LQ_l_b = []
			
			# 针对TINN训练，即随机选择两个相邻帧，
			t_s = random.randint(1, 6)
			LQ_frames_list = [t_s, t_s + 1]

			for v in LQ_frames_list:
				img_GT = util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448))
				img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp_1)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_GT_l.append(img_GT)

			img_GTs = np.stack(img_GT_l, axis=0)
			img_LQs_a = np.stack(img_LQ_l_a, axis=0)
			# BGR to RGB, HWC to CHW, numpy to tensor
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			# img_LQs = img_LQs[:, :, :, [2, 1, 0]]  # LQ是YUV排列的
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()

			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array

			return {'GT': img_GTs, 'LQ_a': img_LQs_a, 'qp_a': qp_1,}
		else:					
			if len(self.opt['qp_list']) == 2:
				qp_1, qp_2 = self.opt['qp_list']
			else:
				# # 随机选择两个qp
				# qp_1, qp_2 = random.sample(self.opt['qp_list'], 2)
    
				# 随机选择一个qp, 另外一个qp也是随机的
				qp_1_id = random.randint(0,4)
				qp_1 = self.opt['qp_list'][qp_1_id]
				if qp_1 == 17:
					offset = 1
				elif qp_1 == 37:
					offset = -1
				else:
					offset = random.randint(0,1)
					if offset == 0:
						offset = -1
				qp_2 = self.opt['qp_list'][qp_1_id + offset]
	
			key = self.paths_GT[index]
			name_a, name_b = key.split('_')
			img_GT_l = []
			img_LQ_l_a = []
			img_LQ_l_b = []
			
			if self.opt['pretrain_ranker']:
				LQ_frames_list = [random.randint(1, 6),]
			else:
				# 针对TINN训练，即随机选择两个相邻帧，
				# LQ_frames_list = self.LQ_frames_list
				t_s = random.randint(1, 6)
				LQ_frames_list = [t_s, t_s + 1]

			for v in LQ_frames_list:
				img_GT = util.read_img(self.GT_env, name_a + '_{}'.format(v), (3, 256, 448))
				img_LQ_l_a.append(util.read_img(self.LQ_env[str(qp_1)], name_a + '_{}'.format(v), (3, 256, 448)))
				img_LQ_l_b.append(util.read_img(self.LQ_env[str(qp_2)], name_a + '_{}'.format(v), (3, 256, 448)))
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
			img_LQs_a = np.stack(img_LQ_l_a, axis=0)
			img_LQs_b = np.stack(img_LQ_l_b, axis=0)
			# BGR to RGB, HWC to CHW, numpy to tensor
			img_GTs = img_GTs[:, :, :, [2, 1, 0]]
			# img_LQs = img_LQs[:, :, :, [2, 1, 0]]  # LQ是YUV排列的
			img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
			img_LQs_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_a, (0, 3, 1, 2)))).float()
			img_LQs_b = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs_b, (0, 3, 1, 2)))).float()
			# return {'LQ': img_LQs, 'GT': img_GTs, 'GT_path': key}

			if self.opt['rgb_2_yuv444p']:
				output_array = np.zeros_like(img_GTs)
				for d in range( img_GTs.shape[0]):
					rgb_image =  rearrange(img_GTs[d], 'c h w -> h w c')
					yuv_image = cv2.cvtColor(rgb_image.numpy(), cv2.COLOR_RGB2YUV)
					output_array[d] = rearrange(yuv_image, 'h w c -> c h w')
				img_GTs = output_array
					
			# # 当预训练ranker时，只用给一帧就行
			# if self.opt['pretrain_ranker']:
			# 	t_c = random.randint(1, 6)
			# 	return {'GT': img_GTs[t_c:t_c+1], 'LQ_a': img_LQs_a[t_c:t_c+1], 'LQ_b': img_LQs_b[t_c:t_c+1], 'qp_a': qp_1, 'qp_b': qp_2}
			# else:
			# 	return {'GT': img_GTs, 'LQ_a': img_LQs_a, 'LQ_b': img_LQs_b, 'qp_a': qp_1, 'qp_b': qp_2}
			return {'GT': img_GTs, 'LQ_a': img_LQs_a, 'LQ_b': img_LQs_b, 'qp_a': qp_1, 'qp_b': qp_2}

	def __len__(self):
		if self.opt['plt_tsne']:
			return 6
		return len(self.paths_GT)
