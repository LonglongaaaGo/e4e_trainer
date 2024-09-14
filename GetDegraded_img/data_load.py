

from PIL import Image
import random
import numpy as np
import cv2
import GetDegraded_img.my_basicsr.my_degradations as degradations
import math


class  Degrader:
    # degrade images
    """docstring for ArtDataset"""
    def __init__(self,im_size=(512,512)):
        super(Degrader, self).__init__()
        # self.root = root
        # self.frame = self._parse_frame()
        # self.transform = transform
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

    # def _parse_frame(self):
    #     frame = []
    #     img_names =[]
    #     listdir(self.root,img_names)
    #     img_names.sort()
    #     for i in range(len(img_names)):
    #         image_path = os.path.join(self.root, img_names[i])
    #         if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
    #             frame.append(image_path)
    #     return frame

    # def __len__(self):
    #     return len(self.frame)

    def degrade_single_path(self,path):
        """
        path: image path
        """
        img = Image.open(path).convert('RGB')
        w,h = img.size


        # w, h = self.im_size[1],self.im_size[0]
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


        # img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        # base_name = os.path.basename(path)
        # target_path = os.path.join(taeget_root,base_name)
        # img_lq.save(target_path)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255)
        return img_lq


    def degrade_single_img(self,img):
        """
        path: PIL image
        """
        # img = Image.open(path).convert('RGB')
        w,h = img.size

        # w, h = self.im_size[1],self.im_size[0]
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


        # img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        # base_name = os.path.basename(path)
        # target_path = os.path.join(taeget_root,base_name)
        # img_lq.save(target_path)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255)
        return img_lq