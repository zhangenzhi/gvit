from gvit.quadtree import QuadTree

import cv2 as cv
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


import os 
import glob 
from pathlib import Path

def get_img_path(datapath="./dataset/exp/"):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files

def get_imagenet_path(datapath="./dataset/"):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files

def transform(img, dsize:tuple=(512, 512)):
    res = cv.resize(img, dsize=dsize, interpolation=cv.INTER_CUBIC)
    grey_img = res[:, :, 0]
    blur = cv.GaussianBlur(grey_img, (3,3),0)
    edge = cv.Canny(blur, 100, 200)
    return res, edge

# save patch sequence
def compress_mix_patches(qdt:QuadTree, img: np.array, to_size:tuple = (8,8,3)):
    h2,w2,c2 = to_size
    seq_patches = qdt.serialize(img)
    for i in range(len(seq_patches)):
        h1, w1, c1 = seq_patches[i].shape
        assert h1==w1, "Need squared input."
        # print(seq_patches[i].shape, seq_patches[i])
        step =int(h1/to_size[0])
        seq_patches[i] = seq_patches[i][::step,::step]
        assert seq_patches[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patches[i].shape, (h2,w2,c2))
    return seq_patches

def custom_partcher(datapath="./dataset/exp", to_size: tuple=(8,8,3)):
    img_path = get_img_path(datapath=datapath)
    patch_size = to_size[0]
    save_to = datapath+"_qdt"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
        
    for i,p in enumerate(img_path):
        img = cv.imread(p)
        img, edge = transform(img)
        qdt = QuadTree(domain=edge)
        seq_patches = compress_mix_patches(qdt, img, to_size)
        seq_img = np.asarray(seq_patches)
        seq_img = np.reshape(seq_img,(patch_size, -1, 3))
        name = Path(p).parts[-2]
        cv.imwrite(save_to+"/{}_{}.jpeg".format(i, name), seq_img)
        
def imagenet_partcher(datapath="./dataset", to_size: tuple=(8,8,3)):
    train_path = os.path.join(datapath, "train")
    val_path = os.path.join(datapath, "val")
    save_to =os.path.join(datapath, "imagenet_qdt")
    if not os.path.exists(save_to):
        os.makedirs(save_to)

def patchify(args):
    datapath = args.path
    if args.dataset == "imagenet":
        imagenet_partcher(datapath=datapath)
    elif args.dataset == "piap":
        pass
    elif args.dataset == "btcv":
        pass
    else:
        custom_patchify()

if __name__ == '__main__':
    args = parser.parse_args()
    