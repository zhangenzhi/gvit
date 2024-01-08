from gvit.quadtree import QuadTree

import torch
import torchvision
from torchvision import transforms
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Patchify dataset.')
parser.add_argument('--dataset', type=str, help='name of the dataset.')
parser.add_argument('--path', type=str, help='path of the dataset.')
parser.add_argument('--max_depth', type=int, default=6, help='path of the dataset.')
parser.add_argument('--max_value', type=int, default=100, help='path of the dataset.')


def compress_mix_patches(qdt:QuadTree, img: np.array, to_size:tuple = (8,8,3)):
    # save patch sequence
    h2,w2,c2 = to_size
    seq_patches = qdt.serialize(img=img[0:512,0:512])
    for i in range(len(seq_patches)):
        h1, w1, c1 = seq_patches[i].shape
        assert h1==w1, "Need squared input."
        # print(seq_patches[i].shape, seq_patches[i])
        step =int(h1/to_size[0])
        seq_patches[i] = seq_patches[i][::step,::step]
        assert seq_patches[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patches[i].shape, (h2,w2,c2))
    return seq_patches

def patchify(args):
    if args.dataset == "imagenet":
        datapath = args.path
        imagenet_partcher(datapath=datapath)
    elif args.dataset == "piap":
        pass
    elif args.dataset == "btcv":
        pass

def imagenet_partcher(datapath):
    train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()  
        ])
    imagenet_data = torchvision.datasets.ImageNet(datapath, transform= train_transform)
    imagenet_val = torchvision.datasets.ImageNet(datapath, split="val", transform= train_transform)
    
if __name__ == '__main__':
    args = parser.parse_args()
    