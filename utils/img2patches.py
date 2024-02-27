import os 
import glob 
from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from gvit.quadtree import QuadTree

# patchify
 
def get_img_path(base="./dataset/exp/"):
    files = []
    for f in glob.glob(os.path.join(base, "*/*.jpeg")):
        files.append(f)
    return files

def transform(img):
    res = cv.resize(img, dsize=(512, 512), interpolation=cv.INTER_CUBIC)
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

def custom_patchify(base="./dataset/exp", to_size: tuple=(8,8,3)):
    img_path = get_img_path(base=base)
    patch_size = to_size[0]
    save_to = base+"_qdt"
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

if __name__ == "__main__":  
    custom_patchify()