#!/usr/bin/python
# -*- encoding: utf-8 -*-

from face_parsing.model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import random
from torch.nn import functional as F


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(out_path='./res/test_res', data_path='./data', ckpt_path='model_final_diss.pth'):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # save_pth = osp.join('res/ckpt_path', ckpt_path)
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(data_path):
            img = Image.open(osp.join(data_path, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(out_path, image_path))


def evaluates_process(out_path='./res/test_res', data_path='./data', ckpt_path='model_final_diss.pth'):
    """
    evaluete the images and save the corresponding map and color maps
    :param out_path:
    :param data_path:
    :param ckpt_path:
    :return:
    """

    os.makedirs(f"{out_path}/semantic_mask",exist_ok=True)
    os.makedirs(f"{out_path}/vis_res",exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # save_pth = osp.join('res/ckpt_path', ckpt_path)
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(data_path):
            img = Image.open(osp.join(data_path, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=f'{out_path}/vis_res/{image_path}')

            cv2.imwrite(f'{out_path}/semantic_mask/{image_path}', parsing)


class Face_parser(torch.nn.Module):

    def __init__(self,device,ckpt_path = "",n_classes = 19):
        super(Face_parser, self).__init__()

        self.n_classes = n_classes
        self.face_parser = BiSeNet(n_classes=n_classes).to(device)
        self.face_parser.load_state_dict(torch.load(ckpt_path))
        self.face_parser.eval()

        self.processing = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.num_of_class = None


    def forward(self, imgs,normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between  [-1,+1] and then scales them between  [0,1]
        If normalize is False, assumes the images are already between  [0,1]
        """
        if normalize:
            imgs = (imgs + 1) * 0.5

        size_ = imgs.shape
        with torch.no_grad():
            imgs = self.processing(imgs)
            imgs = F.interpolate(imgs.detach(), (512, 512), mode="bilinear")
            out = self.face_parser(imgs)[0]
            parse_map = torch.argmax(out, dim=1)
            parse_map = F.interpolate(parse_map.unsqueeze(dim=1).float(), (size_[2], size_[3]), mode="nearest")
            # F.interpolate(aug_mask_01.detach(), (d_segmetation.shape[2], d_segmetation.shape[3]), mode="nearest")
        # parse_map = parse_map.squeeze(1)


        return parse_map

    # def get_color_map(self,parse_map):
    #     # Colors for all 20 parts
    #     part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
    #                    [255, 0, 85], [255, 0, 170],
    #                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
    #                    [0, 255, 85], [0, 255, 170],
    #                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
    #                    [0, 85, 255], [0, 170, 255],
    #                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
    #                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    #
    #     parse_map = parse_map.squeeze(dim=1)
    #     # part_colors = torch.Tensor(part_colors).to(parse_map.device)
    #     vis_parsing_anno_color = np.zeros((parse_map.shape[1], parse_map.shape[2], 3)) + 255
    #
    #     parse_map_cpu = np.array(parse_map.int().squeeze(0).cpu())
    #     num_of_class = torch.max(parse_map.int())
    #     for pi in range(1, num_of_class + 1):
    #         # vis_parsing_anno_color = torch.where(parse_map.squeeze(0) == pi,part_colors[pi].unsqueeze(0), part_colors[0].unsqueeze(0))
    #         index = np.where( parse_map_cpu== pi)
    #         vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    #     return vis_parsing_anno_color


    def get_color_map(self,parse_map):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        # part_colors = np.array(part_colors)
        # np.transpose(part_colors, (1, 2, 0))

        # parse_map = parse_map.squeeze(dim=1)
        part_colors = torch.Tensor(part_colors).to(parse_map.device).unsqueeze(-1).unsqueeze(-1)
        vis_parsing_anno_color = torch.zeros_like(parse_map).repeat([1,3,1,1])+ 222
        # vis_parsing_anno_color = np.zeros((parse_map.shape[1], parse_map.shape[2], 3)) + 255
        # parse_map_cpu = np.array(parse_map.int().squeeze(0).cpu())
        num_of_class = torch.max(parse_map.int())

        for pi in range(0, num_of_class + 1):
            # idxs = (parse_map == pi).nonzero(as_tuple=True)
            # vis_parsing_anno_color[idxs[0],idxs[1],idxs[2],idxs[3]] = part_colors[pi]
            vis_parsing_anno_color = torch.where(parse_map == pi,part_colors[pi], vis_parsing_anno_color)
            # index = np.where( parse_map_cpu== pi)
            # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]


        return vis_parsing_anno_color



    # def get_both_map(self,parse_map,zero_map = None):
    #     """
    #     zero map and semantic map
    #     :param parse_map: semantic map
    #     :return:
    #     """
    #     num_of_class = torch.max(parse_map.int())
    #     idx = random.randint(0,num_of_class)
    #     # idx2 = random.randint(0,num_of_class)
    #
    #     one_map = torch.ones_like(parse_map)
    #     if zero_map == None:
    #         zero_map = torch.zeros_like(parse_map)
    #     zero_map = torch.where(parse_map == idx, one_map, zero_map)
    #     # print(torch.max(zero_map))
    #     return zero_map

    def get_zero_map(self, parse_map, zero_map=None):
        """
        :param parse_map: semantic map
        :return:
        """
        num_of_class = torch.max(parse_map.int())
        # idx = random.randint(0, num_of_class)
        idx = torch.randint(0, num_of_class, (parse_map.shape[0],))
        idx = idx[:,None,None,None].to(parse_map.device)
        # idx2 = random.randint(0,num_of_class)

        one_map = torch.ones_like(parse_map)
        if zero_map == None:
            zero_map = torch.zeros_like(parse_map)
        zero_map = torch.where(parse_map == idx, one_map, zero_map)
        # print(torch.max(zero_map))
        return zero_map


    def get_sparse_semantic_map(self, parse_map):
        """
        :param parse_map:
        :return:
        """
        num_of_class = torch.max(parse_map.int())
        idx = random.randint(0, num_of_class)

        one_map = torch.ones_like(parse_map)
        zero_map = torch.zeros_like(parse_map)
        zero_map = torch.where(parse_map == idx, one_map, zero_map)
        parse_map = parse_map* zero_map
        return parse_map




def get_color_map(parse_map):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # part_colors = np.array(part_colors)
    # np.transpose(part_colors, (1, 2, 0))

    # parse_map = parse_map.squeeze(dim=1)
    part_colors = torch.Tensor(part_colors).to(parse_map.device).unsqueeze(-1).unsqueeze(-1)
    vis_parsing_anno_color = torch.zeros_like(parse_map).repeat([1,3,1,1])+ 222
    # vis_parsing_anno_color = np.zeros((parse_map.shape[1], parse_map.shape[2], 3)) + 255
    # parse_map_cpu = np.array(parse_map.int().squeeze(0).cpu())
    num_of_class = torch.max(parse_map.int())

    for pi in range(0, num_of_class + 1):
        # idxs = (parse_map == pi).nonzero(as_tuple=True)
        # vis_parsing_anno_color[idxs[0],idxs[1],idxs[2],idxs[3]] = part_colors[pi]
        vis_parsing_anno_color = torch.where(parse_map == pi,part_colors[pi], vis_parsing_anno_color)
        # index = np.where( parse_map_cpu== pi)
        # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]


    return vis_parsing_anno_color


def get_zero_map( parse_map, zero_map=None):
    """
    :param parse_map: semantic map
    :return:
    """
    num_of_class = torch.max(parse_map.int())
    # idx = random.randint(0, num_of_class)
    idx = torch.randint(0, num_of_class, (parse_map.shape[0],))
    idx = idx[:,None,None,None].to(parse_map.device)
    # idx2 = random.randint(0,num_of_class)

    one_map = torch.ones_like(parse_map)
    if zero_map == None:
        zero_map = torch.zeros_like(parse_map)
    zero_map = torch.where(parse_map == idx, one_map, zero_map)
    # print(torch.max(zero_map))
    return zero_map


def get_sparse_semantic_map(parse_map):
    """
    :param parse_map:
    :return:
    """
    num_of_class = torch.max(parse_map.int())
    idx = random.randint(0, num_of_class)

    one_map = torch.ones_like(parse_map)
    zero_map = torch.zeros_like(parse_map)
    zero_map = torch.where(parse_map == idx, one_map, zero_map)
    parse_map = parse_map* zero_map
    return parse_map




if __name__ == "__main__":
    evaluate(data_path='./makeup', ckpt_path='79999_iter.pth')


