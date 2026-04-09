'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, shuffle=True, batch_size_preset=None, pin_memory=False):
	phase = dataset_opt['phase']
	if phase == 'train':
		if opt['dist']:
			world_size = torch.distributed.get_world_size()
			num_workers = dataset_opt['n_workers']
			assert dataset_opt['batch_size'] % world_size == 0
			if batch_size_preset is None:
				batch_size = dataset_opt['batch_size'] // world_size
			else:
				batch_size = batch_size_preset
			# shuffle = True
		else:
			num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
			if batch_size_preset is None:
				batch_size = dataset_opt['batch_size']
			else:
				batch_size = batch_size_preset
			# shuffle = True
		return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
										   num_workers=num_workers, sampler=sampler, drop_last=True,
										   pin_memory=pin_memory)
	else:
		return torch.utils.data.DataLoader(dataset, batch_size=opt['datasets']['val']['batch_size'], shuffle=False, num_workers=8,
										   pin_memory=False)

def create_dataloader_compress(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        batch_size = dataset_opt['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                           pin_memory=False)


def create_dataset(dataset_opt):
	mode = dataset_opt['mode']
	if mode == 'LQ':
		# from data.LQ_dataset import LQDataset as D
		pass
	elif mode == 'LQGT':
		from data.LQGT_dataset import LQGTDataset as D
	elif mode == 'Vimeo90K':
		from data.Vimeo90K_dataset import Vimeo90KDataset as D
	elif mode == 'LQGT_Vimeo90K':
		# 两个Vimeo90K dataset, 分别是GT和LQ
		from data.LQGTVimeo90K_dataset import Vimeo90KDataset as D
	elif mode == 'LQGT_Vimeo90K_Pair':
		# Vimeo90K dataset, Pair训练方式
		from data.LQGTVimeo90K_dataset_pair import Vimeo90KDataset as D
	elif mode == 'vimeo_test_septuplet':
		from data.VFI_dataset import VimeoSeptupletDataset as D
	elif mode == 'snu_film_septuplet':
		from data.VFI_dataset import SnuFilmSeptupletDataset as D
	elif mode == 'vimeo_test':
		# from data.video_test_dataset import VideoTestDataset as D
		# from data.Vimeo90K_dataset import Vimeo90KDataset as D
		from data.VFI_dataset import VimeoDataset as D
	# elif mode == 'LQGTseg_bg':
	#     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
	elif mode == 'LQGTVID':
		from data.LQGTVID_dataset import LQGTVIDDataset as D
	else:
		raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

	if mode == 'vimeo_test' or mode == 'vimeo_test_septuplet' or mode == 'snu_film_septuplet':
		dataset = D(dataset_opt['dataroot_GT'])
	else:
		dataset = D(dataset_opt)

	logger = logging.getLogger('base')
	logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
														   dataset_opt['name']))
	return dataset
