import os 
import sys
sys.path.append("./")
import glob 
from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from gvit.quadtree import QuadTree

# patchify
 
def get_png_path(base, resolution):
    data_path = base
    image_filenames = []
    mask_filenames = []

    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
            mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

            # Ensure the image exist
            if os.path.exists(image) and os.path.exists(mask):

                image_filenames.extend([image])
                mask_filenames.extend([mask])
                
    return image_filenames

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

def paip_patchify(base,  resolution: int, to_size: tuple=(8,8,3)):
    img_path = get_png_path(base=base, resolution=resolution)
    patch_size = to_size[0]
    output_dir = base+"_qdt"
    os.makedirs(output_dir, exist_ok=True)
        
    for i,p in enumerate(img_path):
        img = cv.imread(p)
        img, edge = transform(img)
        qdt = QuadTree(domain=edge)
        seq_patches = compress_mix_patches(qdt, img, to_size)
        seq_img = np.asarray(seq_patches)
        seq_img = np.reshape(seq_img,(patch_size, -1, 3))
        name = Path(p).parts[-2]
        cv.imwrite(output_dir+"/{}_{}.jpeg".format(i, name), seq_img)


if __name__ == "__main__":
    # Input directory paths
    image_directory = "/Volumes/data/dataset/paip/output_images_and_masks/"
    resolution = 512
    dataset = "paip"
    if dataset == "paip":
        paip_patchify(base=image_directory, resolution=512)
        print("Patchify finished.")
    else:
        print("No such dataset.")