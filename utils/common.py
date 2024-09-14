from PIL import Image
import matplotlib.pyplot as plt
import torch

# Log images
def log_input_image(x, opts):
	channel_num = x.shape[0]

	if channel_num > 3:
		color_ = torch.tensor([i for i in range(x.shape[0])])*0.05-0.5
		color_ = color_.unsqueeze(-1).unsqueeze(-1).repeat([1,x.shape[1],x.shape[2]]).to(x.device)
		# color_ = torch.randn(x.shape[0],1,1).to(x.device) - 0.5
		# color_ = color_.repeat([1,x.shape[1],x.shape[2]])
		# color_ =(torch.randn(x.shape[0],x.shape[1],x.shape[2]).to(x.device)-0.5)
		x = torch.sum(x*color_,dim=0,keepdim=True).repeat([3,1,1])

	return tensor2im(x)



def semantic2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 6)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_multi_modal_id(hooks_dict, fig, gs, i)
		else:
			# vis_faces_no_id(hooks_dict, fig, gs, i)
			vis_faces_no_id_multi_modal(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def vis_mask_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_mask_modal_id(hooks_dict, fig, gs, i)
		else:
			# vis_faces_no_id(hooks_dict, fig, gs, i)
			# vis_faces_no_id_multi_modal(hooks_dict, fig, gs, i)
			vis_faces_with_mask_modal_id(hooks_dict, fig, gs, i)

	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_with_multi_modal_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))

	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['sketches_img_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 4])
	plt.imshow(hooks_dict['semantic_img_color_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))

	fig.add_subplot(gs[i, 5])
	plt.imshow(hooks_dict['super_color_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))



def vis_faces_with_mask_modal_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))

	# fig.add_subplot(gs[i, 3])
	# plt.imshow(hooks_dict['sketches_img_face'])
	# plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
	# fig.add_subplot(gs[i, 4])
	# plt.imshow(hooks_dict['semantic_img_color_face'])
	# plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
	#
	# fig.add_subplot(gs[i, 5])
	# plt.imshow(hooks_dict['super_color_face'])
	# plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))



def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')


def vis_faces_no_id_multi_modal(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')

	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['sketches_img_face'])
	plt.title('sketches_img_face')
	fig.add_subplot(gs[i, 4])
	plt.imshow(hooks_dict['semantic_img_color_face'])
	plt.title('semantic_img_color_face')
	fig.add_subplot(gs[i, 5])
	plt.imshow(hooks_dict['super_color_face'])
	plt.title('super_color_face')

