from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, CityscapesPathImage
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from scipy import ndimage as ndi



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'doubleattention_resnet50', 'doubleattention_resnet101', 'head_resnet50',
                                 'head_resnet101'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = CityscapesPathImage(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = CityscapesPathImage(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


class PatchManifold(nn.Module):
    def __init__(self, alpha, split, filter):
        super(PatchManifold, self).__init__()
        self.alpha = alpha
        self.split = split
        self.filter = filter

    def forward(self, targets):
        return self.coss_patchmaniflod(targets)

    def expand(self, targets):
        return targets.repeat(1,1,2,2)

    def coss_patchmaniflod(self, targets):
        targets, h , w = self.resample(targets, self.split)

        h = int(h)
        w = int(w)

        sample_h = targets.shape[-2]
        sample_w = targets.shape[-1]

        target_ex = self.expand(targets)

        map_target = torch.zeros([self.split ** 2, self.split **2]).cuda()

        for i in range(self.split):
            for j in range(self.split):
                target = target_ex[:, :, i * h:sample_h + i * h, j * w:sample_w + j * w]

                temp_target = torch.exp(-torch.pow((target - targets), 2))

                for n in range(self.split):
                    for m in range(self.split):
                        pacth_target = temp_target[:, :, n * h:n * h + h, m * w:m * w + w]
                        map_target[n * self.split + m, ((i + n) % self.split) * self.split + (j + m) % self.split] = torch.mean(pacth_target)

        for i in range(self.split * self.split):
            feature = map_target[i]
            feature = torch.sort(feature)
            for j in range(self.split - self.filter):
                ind = feature.indices[j]
                map_target[i,ind] = 0

        return map_target

    def resample(self, targets, split):
        h = targets.shape[-2] / split
        w = targets.shape[-1] / split

        h = int(h / 10)
        w = int(w / 10)

        reshape_h = h * split
        reshape_w = w * split

        x = F.interpolate(targets, size=[int(reshape_h), int(reshape_w)], mode='bilinear', align_corners=False)

        return x, h, w


def matrix_to_dictionary(matrix):
    (h,w) = matrix.shape
    list = []
    for i in range(h):
        for j in range(w):
            if matrix[i,j] != 0:
                dict = {'i':i,'j':j,'value':matrix[i,j]}
                list.append(dict)
    return list


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    if opts.dataset.lower == 'cityscapes':
        opts.num_classes = 19

    # setup cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' %device)

    # set up dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True)
    print('Dataset: %s, Train set: %d, val set: %d' %(opts.dataset, len(train_dst), len(val_dst)))

    split_m = [15,20,25]
    for split_n in split_m:
        patchmanifold = PatchManifold(alpha=1, split=split_n, filter=3)

        with tqdm(total=len(train_dst),  unit='img') as pbar:
            for i, dataset in enumerate(train_loader):
                image_path = dataset['image_path']
                target_path = dataset['target_path']
                image = dataset['image']
                # print(image.shape)
                # print(image_path)
                # print(target_path)

                weights_manifold = patchmanifold(image)


                break

                save_path = image_path[0].split('/')
                # print(save_path)
                weights_name = '{}_{}'.format(save_path[8].split('_leftImg8bit')[0],'weights.pth')
                save_path = os.path.join('/', save_path[1], save_path[2], save_path[3], save_path[4],\
                                         'weights_{}'.format(split_n), save_path[6], save_path[7])

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    # if os.path.exists(save_path):
                    #     print('create folder {}'.format(save_path))
                # print(save_path)
                weights_name = os.path.join(save_path, weights_name)

                torch.save(weights_manifold,weights_name)
                pbar.update(image.shape[0])

        #     # test = torch.load(weights_name)
        #     # print(save_path)
        # with tqdm(total=len(val_dst),  unit='img') as pbar:
        #     for i, dataset in enumerate(val_loader):
        #         image_path = dataset['image_path']
        #         target_path = dataset['target_path']
        #         image = dataset['image']
        #         # print(image.shape)
        #         # print(image_path)
        #         # print(target_path)
        #
        #         weights_manifold = patchmanifold(image)
        #         save_path = image_path[0].split('/')
        #         # print(save_path)
        #         weights_name = '{}_{}'.format(save_path[8].split('_leftImg8bit')[0],'weights.pth')
        #         save_path = os.path.join('/', save_path[1], save_path[2], save_path[3], save_path[4],\
        #                                  'weights_{}'.format(split_n), save_path[6], save_path[7])
        #
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #             # if os.path.exists(save_path):
        #             #     print('create folder {}'.format(save_path))
        #         # print(save_path)
        #         weights_name = os.path.join(save_path, weights_name)
        #
        #         torch.save(weights_manifold,weights_name)
        #         pbar.update(image.shape[0])
        #
        #     # test = torch.load(weights_name)
        #     # print(save_path)


if __name__ == '__main__':
    main()