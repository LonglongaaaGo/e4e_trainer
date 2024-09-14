from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
from GetDegraded_img.data_load import Degrader
import my_basicsr.my_degradations as degradations
import math


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.img_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		print("images:",len(self.img_paths))

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		from_path = self.img_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im



def erode_demo(e_image):
    kernel_size = random.randint(3,7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))  # 定义结构元素的形状和大小  矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # 十字形
    image = cv2.erode(e_image, kernel)  # 腐蚀操作
    # plt_show_Image_image(image)
    return image
    # 腐蚀主要就是调用cv2.erode(img,kernel,iterations)，这个函数的参数是
    # 第一个参数：img指需要腐蚀的图
    # 第二个参数：kernel指腐蚀操作的内核，默认是一个简单的3X3矩阵，我们也可以利用getStructuringElement（）函数指明它的形状
    # 第三个参数：iterations指的是腐蚀次数，省略是默认为1




class  ImageFolder_restore(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        # #
        self.blur_kernel_size = [19, 20]
        self.kernel_list = ('iso', 'aniso')
        self.kernel_prob= [0.5, 0.5]
        self.blur_sigma= [0.1, 10]
        self.downsample_range= [0.8, 8]
        self.noise_range=[0, 20]
        self.jpeg_range= [60, 100]

        self.color_jitter_prob= None
        self.color_jitter_shift=20
        self.color_jitter_pt_prob= None
        self.gray_prob= None
        self.gt_gray= True

    def _parse_frame(self):
        frame = []
        img_names =[]
        listdir(self.root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        w, h = self.im_size[1],self.im_size[0]
        img_gt = np.array(img).copy().astype(np.float32)/255.0
        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # if self.transform:
        #     img = self.transform(img)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        img_gt = np.array(img)
        img_gt = np.ascontiguousarray(img_gt.transpose(2, 0, 1))  # HWC => CHW
        img_lq = np.ascontiguousarray(img_lq.transpose(2, 0, 1))  # HWC => CHW

        return img_lq,img_gt





class  ImageFolder_restore_free_form(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore_free_form, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        # #
        self.blur_kernel_size = [19, 20] #ok
        self.kernel_list = ('iso', 'aniso') #ok
        self.kernel_prob= [0.5, 0.5]  #ok
        self.blur_sigma= [0.1, 10]  #ok
        self.downsample_range= [0.8, 8]  #ok
        self.noise_range=[0, 20]  #ok
        self.jpeg_range= [60, 100]  #ok

        self.color_jitter_prob= None  #ok
        self.color_jitter_shift=20  #ok
        self.color_jitter_pt_prob= None  #ok
        self.gray_prob= 0.008  #ok
        self.gt_gray= True  #ok
        self.hazy_prob = 0.008  #ok
        self.hazy_alpha = [0.75, 0.95]  #ok

        # self.shift_prob =  0.2  #ok
        self.shift_prob =  0  #ok
        self.shift_unit = 1  #ok
        self.shift_max_num = 32  #ok

    def _parse_frame(self):
        frame = []
        img_names =[]
        listdir(self.root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        # RandomHorizontalFlip
        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        w, h = self.im_size[1],self.im_size[0]
        img_gt = np.array(img).copy().astype(np.float32)/255.0

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob):
            # self.shift_unit = 32
            # import pdb
            # pdb.set_trace()
            shift_vertical_num = np.random.randint(0, self.shift_max_num * 2 + 1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num * 2 + 1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit),
                                         (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit),
                                         (0, 0)),
                                mode='symmetric')
            img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w, :]

        img_lq1 = self.degrade_img(img_gt)
        img_lq2 = self.degrade_img(img_gt)

        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq1 = cv2.cvtColor(img_lq1, cv2.COLOR_BGR2GRAY)
            img_lq2 = cv2.cvtColor(img_lq2, cv2.COLOR_BGR2GRAY)

            img_lq1 = np.tile(img_lq1[:, :, None], [1, 1, 3])
            img_lq2 = np.tile(img_lq2[:, :, None], [1, 1, 3])

            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])


        # img_gt = np.array(img)

        img_gt = np.ascontiguousarray(img_gt.transpose(2, 0, 1))  # HWC => CHW
        img_lq1 = np.ascontiguousarray(img_lq1.transpose(2, 0, 1))  # HWC => CHW
        img_lq2 = np.ascontiguousarray(img_lq2.transpose(2, 0, 1))  # HWC => CHW

        return img_lq1,img_lq2,img_gt


    def degrade_img(self,img_gt):
        w, h = self.im_size[1],self.im_size[0]

        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        ## add simple hazy
        if (self.hazy_prob is not None) and (np.random.uniform() < self.hazy_prob):
            alpha = np.random.uniform(self.hazy_alpha[0], self.hazy_alpha[1])
            img_lq = img_lq * alpha + np.ones_like(img_lq) * (1 - alpha)

        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)


        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # if self.transform:
        #     img = self.transform(img)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        return img_lq



class MultiImagesDataset(Dataset):
	"""
	Multi model editing
	: sketches
	: semantics
	: colors
	"""

	def __init__(self, img_root,semantic_root, edge_root,color_root,
				 opts, target_transform=None,
				 im_size=(256, 256) ):

		self.source_paths = sorted(data_utils.make_dataset(img_root))
		self.semantic_paths = sorted(data_utils.make_dataset(semantic_root))
		self.edge_paths = sorted(data_utils.make_dataset(edge_root))
		self.color_paths = sorted(data_utils.make_dataset(color_root))

		# self.target_paths = sorted(data_utils.make_dataset(target_root))
		# self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		print("images:",len(self.source_paths))
		self.im_size = im_size

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		img_path = self.source_paths[index]
		semantic_path = self.semantic_paths[index]
		edge_path = self.edge_paths[index]
		color_path = self.color_paths[index]

		img = Image.open(img_path)
		img = img.convert('RGB')
		edge_img = Image.open(edge_path).convert('L')
		semancti_map = Image.open(semantic_path).convert('L')

		color_img = Image.open(color_path).convert('RGB')

		w, h = img.size
		edge_img = edge_img.resize((w, h), Image.NEAREST)
		semancti_map = semancti_map.resize((w, h), Image.NEAREST)
		color_img = color_img.resize((w, h), Image.NEAREST)
		if h != self.im_size[0] or w != self.im_size[1]:
			ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
			new_w = int(ratio * w)
			new_h = int(ratio * h)
			img_scaled = img.resize((new_w, new_h), Image.ANTIALIAS)
			edge_img_scaled = edge_img.resize((new_w, new_h), Image.NEAREST)
			semancti_map_scaled = semancti_map.resize((new_w, new_h), Image.NEAREST)
			color_img_scaled = color_img.resize((new_w, new_h), Image.ANTIALIAS)

			h_rang = new_h - self.im_size[0]
			w_rang = new_w - self.im_size[1]
			h_idx = 0
			w_idx = 0
			if h_rang > 0: h_idx = random.randint(0, h_rang)
			if w_rang > 0: w_idx = random.randint(0, w_rang)
			#
			img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
			edge_img = edge_img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
			color_img = color_img_scaled.crop(
				(w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
			semancti_map = semancti_map_scaled.crop(
				(w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

		# RandomHorizontalFlip
		flip = random.randint(0, 1)
		if flip == 1:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			edge_img = edge_img.transpose(Image.FLIP_LEFT_RIGHT)
			semancti_map = semancti_map.transpose(Image.FLIP_LEFT_RIGHT)
			color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)

		#
		edge_img = np.array(edge_img)
		if edge_img.ndim == 2:
			edge_img = np.expand_dims(edge_img, axis=0)
		else:
			edge_img = edge_img[0:1, :, :]
		edge_img[edge_img < 128] = 1.0
		edge_img[edge_img >= 128] = 0
		#
		rand_ = random.randint(1, 2)
		# mask = erode_demo(mask)
		if rand_ == 1:
			edge_img = erode_demo(edge_img)

		# semantic
		# semancti_map = np.array(semancti_map)
		# semancti_map = torch.from_numpy(semancti_map).unsqueeze(0)
		# semantic
		semancti_map = np.array(semancti_map)
		semancti_map = torch.from_numpy(semancti_map)
		semancti_map2d = semancti_map.clone().unsqueeze(0)

		semancti_map = F.one_hot(semancti_map.to(torch.int64), num_classes=19)
		semancti_map = semancti_map.permute(2, 0, 1)

		#
		if self.target_transform:
			img = self.target_transform(img)
			color_img = self.target_transform(color_img)

		return img,semancti_map2d,edge_img, color_img,semancti_map





class MaskImageDataset(Dataset):
	"""
	Multi model editing
	"""

	def __init__(self, img_root,condition_root,opts, target_transform=None,
				 im_size=(256, 256),condition_type = "image" ):

		self.source_paths = sorted(data_utils.make_dataset(img_root))
		self.condition_paths = sorted(data_utils.make_dataset(condition_root))

		# self.semantic_paths = sorted(data_utils.make_dataset(semantic_root))
		# self.edge_paths = sorted(data_utils.make_dataset(edge_root))
		# self.color_paths = sorted(data_utils.make_dataset(color_root))

		# self.target_paths = sorted(data_utils.make_dataset(target_root))
		# self.source_transform = source_transform

		self.target_transform = target_transform
		self.opts = opts
		print("images:",len(self.source_paths))
		self.im_size = im_size
		# which type you will use
		self.condition_type = condition_type
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		img_path = self.source_paths[index]
		condition_path = self.condition_paths[index]

		# semantic_path = self.semantic_paths[index]
		# edge_path = self.edge_paths[index]
		# color_path = self.color_paths[index]

		img = Image.open(img_path).convert('RGB')

		if self.condition_type == "image":
			condition_img = Image.open(condition_path).convert('RGB')
		elif self.condition_type == "hed_sketch":
			condition_img = Image.open(condition_path).convert('RGB')
		elif self.condition_type == "sketch":
			condition_img = Image.open(condition_path).convert('L')
		elif self.condition_type == "semantic":
			condition_img = Image.open(condition_path).convert('L')


		# edge_img = Image.open(edge_path).convert('L')
		# semancti_map = Image.open(semantic_path).convert('L')
		# color_img = Image.open(color_path).convert('RGB')

		w, h = img.size

		cw,ch = condition_img.size
		# edge_img = edge_img.resize((w, h), Image.NEAREST)
		# semancti_map = semancti_map.resize((w, h), Image.NEAREST)
		# color_img = color_img.resize((w, h), Image.NEAREST)
		# if h != self.im_size[0] or w != self.im_size[1] or ch != self.im_size[0] or cw != self.im_size[1]:
		# 	ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
		# 	new_w = int(ratio * w)
		# 	new_h = int(ratio * h)
		# 	img_scaled = img.resize((new_w, new_h), Image.ANTIALIAS)
		# 	condition_img_scaled = condition_img.resize((new_w, new_h), Image.NEAREST)
		#
		# 	# edge_img_scaled = edge_img.resize((new_w, new_h), Image.NEAREST)
		# 	# semancti_map_scaled = semancti_map.resize((new_w, new_h), Image.NEAREST)
		# 	# color_img_scaled = color_img.resize((new_w, new_h), Image.ANTIALIAS)
		#
		# 	h_rang = new_h - self.im_size[0]
		# 	w_rang = new_w - self.im_size[1]
		# 	h_idx = 0
		# 	w_idx = 0
		# 	if h_rang > 0: h_idx = random.randint(0, h_rang)
		# 	if w_rang > 0: w_idx = random.randint(0, w_rang)
		# 	#
		# 	img = img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
		# 	condition_img = condition_img_scaled.crop(
		# 		(w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
		#
		# 	# edge_img = edge_img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
		# 	# color_img = color_img_scaled.crop(
		# 	# 	(w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))
		# 	# semancti_map = semancti_map_scaled.crop(
		# 	# 	(w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

		# RandomHorizontalFlip
		flip = random.randint(0, 1)
		if flip == 1:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			condition_img = condition_img.transpose(Image.FLIP_LEFT_RIGHT)
			# edge_img = edge_img.transpose(Image.FLIP_LEFT_RIGHT)
			# semancti_map = semancti_map.transpose(Image.FLIP_LEFT_RIGHT)
			# color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)

		if self.condition_type == "sketch":
			condition_img = np.array(condition_img)
			if condition_img.ndim == 2:
				condition_img = np.expand_dims(condition_img, axis=0)
			else:
				condition_img = condition_img[0:1, :, :]
			condition_img[condition_img < 128] = 1.0
			condition_img[condition_img >= 128] = 0
			rand_ = random.randint(1, 2)
			# mask = erode_demo(mask)
			if rand_ == 1:
				condition_img = erode_demo(condition_img)
		elif self.condition_type == "semantic":
			# semantic
			condition_img = np.array(condition_img)
			condition_img = torch.from_numpy(condition_img).unsqueeze(0)
			# semantic
			# condition_img = np.array(condition_img)
			# condition_img = torch.from_numpy(condition_img)
			# condition_img2d = condition_img.clone().unsqueeze(0)

			condition_img = F.one_hot(condition_img.to(torch.int64), num_classes=19).squeeze(0)
			condition_img = condition_img.permute(2, 0, 1)

		elif self.condition_type == "hed_sketch":
			# rand_ = random.randint(1, 2)
			# if rand_ == 1:
			# 	condition_img = erode_demo(condition_img)
			condition_img = self.target_transform(condition_img)

		elif self.condition_type == "image":
			condition_img = self.target_transform(condition_img)

		#source image
		if self.target_transform:
			img = self.target_transform(img)

		return img,condition_img




def listdir(path, list_name):  # 传入存储的list
    '''
    递归得获取对应文件夹下的所有文件名的全路径
    存在list_name 中
    :param path: input the dir which want to get the file list
    :param list_name:  the file list got from path
	no return
    '''
    list_dirs = os.listdir(path)
    list_dirs.sort()
    for file in list_dirs:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    list_name.sort()
    # print(list_name)


class ImageFolder_semantic_edges_color_v3(Dataset):
    """
    load images and edge maps
    add color map
    add semantic map  to one hot of semantic images
    # add more modals
    """
    def __init__(self, image_root,
                 semantic_root,
                 edge_root,
                 canny_root,
                 landmark_root,
                 hed_root,
                 depth_root,
                 midas_root,
                 geometry_root,
                 color_root,
                 gray_transform=None,
                 color_transform=None,
                 im_size=(256,256)
				 ):
        """
        :param image_root: root for the images
        :param edge_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_semantic_edges_color_v3, self).__init__()
        self.image_root = image_root

        self.edge_root = edge_root  #binary
        self.canny_root = canny_root #binary
        self.landmark_root = landmark_root #binary

        self.hed_root = hed_root  #gray
        self.depth_root = depth_root #gray
        self.midas_root = midas_root # gray
        self.geometry_root = geometry_root

        self.color_root = color_root # color

        self.semantic_root = semantic_root # semantic

        self.frame = self._parse_frame(self.image_root)

        self.edge_frame = self._parse_frame(self.edge_root)
        self.canny_frame = self._parse_frame(self.canny_root)
        self.landmark_frame = self._parse_frame(self.landmark_root)
        self.hed_frame = self._parse_frame(self.hed_root)
        self.depth_frame = self._parse_frame(self.depth_root)
        self.color_frame = self._parse_frame(self.color_root)
        self.midas_frame = self._parse_frame(self.midas_root)
        self.geometry_frame = self._parse_frame(self.geometry_root)

        self.semantic_frame = self._parse_frame(self.semantic_root)

        self.degrader = Degrader(im_size=im_size)

        self.color_transform = color_transform
        self.gray_transform = gray_transform

        self.im_size = im_size

    def _parse_frame(self,root):
        img_names = []
        listdir(root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)


    def binary_fun(self,edge_img,reverse= False):
        """
        binary function
        reverse : which direction? 255 = 0? or 255 =1 ?
        :return:
        """

        edge_img = np.array(edge_img)
        if edge_img.ndim == 2:
            mask = np.expand_dims(edge_img, axis=0)
        else:
            mask = edge_img[0:1, :, :]

        if reverse:
            mask[mask < 128] = 0.0
            mask[mask >= 128] = 1.0
        else:
            mask[mask < 128] = 1.0
            mask[mask >= 128] = 0

        rand_ = random.randint(1, 2)
        if rand_ == 1:
            mask = erode_demo(mask)
        return mask

    def __getitem__(self, idx):
        file = self.frame[idx]
        edge_file = self.edge_frame[idx]
        canny_file = self.canny_frame[idx]
        landmark_file = self.landmark_frame[idx]

        hed_file = self.hed_frame[idx]
        depth_file = self.depth_frame[idx]
        midas_file = self.midas_frame[idx]
        geometry_file = self.geometry_frame[idx]

        color_file = self.color_frame[idx]
        semantic_file = self.semantic_frame[idx]

        img = Image.open(file).convert('RGB')

        edge_img = Image.open(edge_file).convert('L')
        canny_img = Image.open(canny_file).convert('L')
        landmark_img = Image.open(landmark_file).convert('L')

        hed_img = Image.open(hed_file).convert('L')
        depth_img = Image.open(depth_file).convert('L')
        midas_img = Image.open(midas_file).convert('L')
        geometry_img = Image.open(geometry_file).convert('L')

        color_img = Image.open(color_file).convert('RGB')

        semancti_map = Image.open(semantic_file).convert('L')

        w,h = img.size
        edge_img = edge_img.resize((w,h),Image.NEAREST)
        canny_img = canny_img.resize((w,h),Image.NEAREST)
        landmark_img = landmark_img.resize((w,h),Image.NEAREST)

        hed_img = hed_img.resize((w,h),Image.BILINEAR)
        depth_img = depth_img.resize((w,h),Image.BILINEAR)
        midas_img = midas_img.resize((w,h),Image.BILINEAR)
        geometry_img = geometry_img.resize((w,h),Image.BILINEAR)

        # degrade_img = degrade_img.resize((w,h),Image.BILINEAR)
        color_img = color_img.resize((w,h),Image.BILINEAR)

        semancti_map = semancti_map.resize((w,h),Image.NEAREST)

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)

            edge_img_scaled = edge_img.resize((new_w,new_h),Image.NEAREST)
            canny_img_scaled = canny_img.resize((new_w,new_h),Image.NEAREST)
            landmark_img_scaled = landmark_img.resize((new_w,new_h),Image.NEAREST)

            hed_img_scaled = hed_img.resize((new_w,new_h),Image.BILINEAR)
            depth_img_scaled = depth_img.resize((new_w,new_h),Image.BILINEAR)
            midas_img_scaled = midas_img.resize((new_w,new_h),Image.BILINEAR)
            geometry_img_scaled = geometry_img.resize((new_w,new_h),Image.BILINEAR)

            color_img_scaled = color_img.resize((new_w, new_h), Image.ANTIALIAS)
            # degrade_img_scaled = degrade_img.resize((new_w, new_h), Image.ANTIALIAS)

            semancti_map_scaled = semancti_map.resize((new_w,new_h),Image.NEAREST)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            #
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            edge_img = edge_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            canny_img = canny_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            landmark_img = landmark_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

            hed_img = hed_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            depth_img = depth_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            midas_img = midas_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            geometry_img = geometry_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

            # degrade_img = degrade_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            color_img = color_img_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))

            semancti_map = semancti_map_scaled.crop((w_idx, h_idx, int(w_idx + self.im_size[1]), int(h_idx + self.im_size[0])))


        # RandomHorizontalFlip
        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            edge_img = edge_img.transpose(Image.FLIP_LEFT_RIGHT)
            canny_img = canny_img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark_img = landmark_img.transpose(Image.FLIP_LEFT_RIGHT)

            hed_img = hed_img.transpose(Image.FLIP_LEFT_RIGHT)
            depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
            midas_img = midas_img.transpose(Image.FLIP_LEFT_RIGHT)
            geometry_img = geometry_img.transpose(Image.FLIP_LEFT_RIGHT)

            color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)
            # degrade_img = degrade_img.transpose(Image.FLIP_LEFT_RIGHT)

            semancti_map = semancti_map.transpose(Image.FLIP_LEFT_RIGHT)

        edge_img = self.binary_fun(edge_img)
        canny_img = self.binary_fun(canny_img,reverse=True)
        landmark_img = self.binary_fun(landmark_img,reverse=True)

        if self.gray_transform:
            hed_img = self.gray_transform(hed_img)
            depth_img = self.gray_transform(depth_img)
            midas_img = self.gray_transform(midas_img)
            geometry_img = self.gray_transform(geometry_img)

        # degrade
        degrade_img = self.degrader.degrade_single_img(img.copy())
        degrade_img = np.ascontiguousarray(degrade_img.transpose(2, 0, 1))  # HWC => CHW
        degrade_img = (degrade_img.astype(np.float32) / 127.5) - 1.0

        if self.color_transform:
            img = self.color_transform(img)
            color_maps = self.color_transform(color_img)
            # degrade_img = self.color_transform(degrade_img)



        # semantic
        semancti_map = np.array(semancti_map)
        #
        semancti_map = torch.from_numpy(semancti_map)
        semancti_map2d = semancti_map.clone().unsqueeze(0)

        semancti_map = F.one_hot(semancti_map.to(torch.int64), num_classes=19)
        semancti_map = semancti_map.permute(2, 0, 1)
        # semancti_map = torch.from_numpy(semancti_map.transpose((2, 0, 1)))


        return img,edge_img,canny_img,landmark_img,hed_img,depth_img,midas_img,geometry_img,color_maps, degrade_img, semancti_map2d,semancti_map


