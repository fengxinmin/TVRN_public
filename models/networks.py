import logging
import math

from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet
try:
	from models.modules.CSTVR_Net import *
except:
	print("failed to load CSTVR...")

logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt, case=None, fixed_QP=True, ranker_inchans=None, ranker_outchans=1):
	opt_net = opt['network_G']
	which_model = opt_net['which_model_G']
	subnet_type = which_model['subnet_type']
	opt_datasets = opt['datasets']

	if opt_net['init']:
		init = opt_net['init']
	else:
		init = 'xavier'
	down_num = int(math.log(opt_net['scale'], 2))
	if opt['scale_type'] == 'temporal':
		netG = CodecTemporalNet(opt, subnet(subnet_type, init), down_num)
	elif opt['scale_type'] == 'CSTVR_w_surrogate_net':
		netG = CSTVR_w_surrogate_net(opt, subnet(subnet_type, init), down_num)
	return netG 
