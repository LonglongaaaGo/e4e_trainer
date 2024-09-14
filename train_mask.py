"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")
sys.path.append('../training')

from options.train_options import TrainOptions
from training.Mask_coach import  Coach
### traing encodings for the warping



"""
orignal training codes
+ masked image training
"""

def main():
	opts = TrainOptions().parse()
	previous_train_ckpt = None
	if opts.resume_training_from_ckpt:
		opts, previous_train_ckpt = load_train_checkpoint(opts)
	else:
		setup_progressive_steps(opts)
		create_initial_experiment_dir(opts)

	coach = Coach(opts, previous_train_ckpt)
	coach.train()


def load_train_checkpoint(opts):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = vars(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = Namespace(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt


def setup_progressive_steps(opts):
	log_size = int(math.log(opts.stylegan_size, 2))
	num_style_layers = 2*log_size - 2
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir,exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
	for k, v in new_opts.items():
		if k not in ckpt_opts:
			ckpt_opts[k] = v
	if new_opts['update_param_list']:
		for param in new_opts['update_param_list']:
			ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
	main()



"""
--dataset_type
ffhq_encode
--exp_dir
experiment
--start_from_latent_avg
--use_w_pool
--w_discriminator_lambda
0.1
--progressive_start
20000
--id_lambda
0.5
--val_interval
10000
--max_steps
200000
--stylegan_size
1024
--workers
8
--batch_size
2
--test_batch_size
4
--test_workers
4
--save_training_data
--keep_optimizer
--in_channel
19
--stylegan_weights
/home/iccv/workspace/Longlongaaago/Pre-trained/StyleGAN_rosinality/stylegan2-ffhq-config-f.pt
--ir_se50_path
/home/iccv/workspace/Longlongaaago/Pre-trained/model_ir_se50.pth
--img_path
/home/iccv/workspace/Data/ffhq_256_anno/test_256
--condition_type
semantic
--condition_path
/home/iccv/workspace/Data/ffhq_semantic/test_semantic_mask
--multi_modal
True
"""