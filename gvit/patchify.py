import sys
sys.path.append("./")

from gvit.quadtree import QuadTree

import cv2 as cv
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from utils.savescsv import savedict, savelist
import argparse

import os 
import glob 
from pathlib import Path

def get_img_path(datapath):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files

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

def get_imagenet_path(datapath):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files

def plot_img_patch_dist(patches_info):
    length = [sum(img_dict.values()) for img_dict in patches_info]
    
    savelist(length, "length")
    # Plotting
    plt.hist(length, color='red', edgecolor='black')
    plt.xlabel('Sequence Length', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Distribution of Patches by PAIP Samples', fontsize=18)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()

    # Save the figure
    plt.savefig('histogram_plot.pdf')
    plt.close()
    
    
def plot_patchied_info(patches_info):
    print(patches_info)
    savedict(patches_info,name="patches")
    # Extract keys and values
    keys = list(patches_info.keys())
    values = list(patches_info.values())
    
    # And sort keys in ascending order
    keys_sorted = sorted(keys, key=lambda x: int(x.split('*')[0]))
    # Arrange values in the corresponding order
    values_sorted = [patches_info[k]/208 for k in keys_sorted]
    keys_sorted = [x.split('*')[0] for x in keys_sorted]


    # Plotting
    plt.bar(keys_sorted, values_sorted, color='blue')
    plt.xlabel('Size of Patches', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.title('Count of Total Patches of PAIP', fontsize=18)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('bar_plot.pdf')
    plt.close()
    return sum(values_sorted)
    
def transform(img, sth:int=3, canny:tuple=(100,200), dsize:tuple=(512, 512)):
    res = cv.resize(img, dsize=dsize, interpolation=cv.INTER_CUBIC)
    grey_img = res[:, :, 0]
    blur = cv.GaussianBlur(grey_img, (sth,sth), 0)
    edge = cv.Canny(blur, canny[0], canny[1])
    return res, edge

def count_info(qdt:QuadTree):
    patch_info = {}
    print(qdt.count_patches(patch_info))
    print(patch_info)
    print(sum(patch_info.values()))
    return patch_info

# save patch sequence
def compress_mix_patches(qdt:QuadTree, img: np.array, to_size:tuple=(8,8,3), target_length = 576):
    
    seq_patches = qdt.serialize(img)
    patch_info = {}
    qdt.count_patches(patch_info)
    # min_key = min((tuple(map(int, key.split('*'))), value) for key, value in patch_info.items())[0]
    # to_size = list(min_key) + [3]
    h2,w2,c2 = to_size
    
    for i in range(len(seq_patches)):
        h1, w1, c1 = seq_patches[i].shape
        assert h1==w1, "Need squared input."
        seq_patches[i] = cv.resize(seq_patches[i], (h2, w2), interpolation=cv.INTER_CUBIC)
        assert seq_patches[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patches[i].shape, (h2,w2,c2))
        
    original_length = len(seq_patches)

    if original_length > target_length:
        # Randomly drop patches to achieve the target length
        indices_to_keep = np.random.choice(original_length, target_length, replace=False)
        seq_patches = [seq_patches[i] for i in indices_to_keep]
    elif original_length < target_length:
        # Pad patches to achieve the target length
        num_padding = target_length - original_length
        pad_indices = np.random.choice(original_length, num_padding, replace=True)
        padded_patches = [seq_patches[i] for i in pad_indices]
        seq_patches.extend(padded_patches)
        
    return seq_patches, to_size, patch_info

import multiprocessing as mp
def ppatch(path, output_dir, target_length, sth, resolution, split_value, max_depth, to_size):
    img = cv.imread(path)
    img, edge = transform(img, sth=sth, dsize=(resolution, resolution))
    qdt = QuadTree(domain=edge, max_value=split_value, max_depth = max_depth)
    seq_patches, patch_size, patch_info = compress_mix_patches(qdt, img, to_size, target_length)
    seq_img = np.asarray(seq_patches)
    seq_img = np.reshape(seq_img, [patch_size[0], -1, patch_size[2]])
    name = Path(path).parts[-2]
    cv.imwrite(output_dir+"/{}/{}_{}-p.png".format(name, resolution, "qdt"), seq_img)
    return patch_info

result_list = []
def paip_patchify(base, split_value:int, max_depth:int, resolution: int, sth:int=3, target_length:int=576, to_size:tuple=(8,8,3)):
    img_path = get_png_path(base=base, resolution=resolution)
    output_dir = base
    os.makedirs(output_dir, exist_ok=True)
    total_patches_info = []
    statical_info = {}
    
    def log_result(result):
        result_list.append(result)
    pool = mp.Pool(8)
    for p in img_path:
        pool.apply_async(func=ppatch, 
                         args=(p, output_dir, target_length, sth, resolution, split_value, max_depth, to_size, ),
                         callback = log_result)
 
    pool.close()
    pool.join()
    
    total_patches_info = result_list
    # statical calculate
    for pi in result_list:
        for key, value in pi.items():
            if key in statical_info:
                statical_info[key] += value
            else:
                statical_info[key] = value
                
    avg_len = plot_patchied_info(statical_info)
    plot_img_patch_dist(total_patches_info)
    print("Avg lenth:{}, resolution:{}, to_size:{}, sp_val:{}, sth:{}".format(avg_len, resolution, to_size[0], split_value, sth))

# def paip_patchify(base, split_value:int, max_depth:int, resolution: int, sth:int=3, target_length:int=576, to_size:tuple=(8,8,3)):
#     img_path = get_png_path(base=base, resolution=resolution)
#     output_dir = base
#     os.makedirs(output_dir, exist_ok=True)
#     total_patches_info = []
#     statical_info = {}
#     for i,p in enumerate(img_path):
#         img = cv.imread(p)
#         img, edge = transform(img, sth=sth, dsize=(resolution, resolution))
#         qdt = QuadTree(domain=edge, max_value=split_value, max_depth = max_depth)
#         seq_patches, patch_size, patch_info = compress_mix_patches(qdt, img, to_size, target_length)
#         seq_img = np.asarray(seq_patches)
#         seq_img = np.reshape(seq_img, [patch_size[0], -1, patch_size[2]])
#         name = Path(p).parts[-2]
#         cv.imwrite(output_dir+"/{}/{}_{}.png".format(name, resolution, "qdt"), seq_img)
#         total_patches_info.append(patch_info)
#         # statical calculate
#         for key, value in patch_info.items():
#             if key in statical_info:
#                 statical_info[key] += value
#             else:
#                 statical_info[key] = value
                
#     avg_len = plot_patchied_info(statical_info)
#     plot_img_patch_dist(total_patches_info)
#     print("Avg lenth:{}, resolution:{}, to_size:{}, sp_val:{}, sth:{}".format(avg_len, resolution, to_size[0], split_value, sth))
        
def imagenet_patcher(datapath):
    train_path = os.path.join(datapath, "train")
    val_path = os.path.join(datapath, "val")
    save_to =os.path.join(datapath, "imagenet_qdt")
    if not os.path.exists(save_to):
        os.makedirs(save_to)

def patchify(args):
    datapath = args.datapath
    if args.dataset == "imagenet":
        imagenet_patcher(datapath=datapath)
    elif args.dataset == "paip":
        paip_patchify(base=datapath, 
                      resolution=args.resolution,
                      sth=args.sth,
                      split_value=args.split_value,
                      target_length=args.target_length,
                      max_depth=args.max_depth,
                      to_size=(args.to_size, args.to_size, 3))
    elif args.dataset == "btcv":
        pass
    else:
        pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Patchify dataset.')
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--resolution', type=int, default=512, help='resolution of the img.')
    parser.add_argument('--max_depth', type=int, default=10, help='path of the dataset.')
    parser.add_argument('--to_size', type=int, default=8, help='path of the dataset.')
    parser.add_argument('--target_length', type=int, default=576, help='path of the dataset.')
    parser.add_argument('--sth', type=int, default=3, help='smooth factor for gaussain smoothing.')
    parser.add_argument('--split_value', type=int, default=50, help='criteron value to subdivision.')
    parser.add_argument('--datapath',  type=str, default="/Volumes/data/dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    args = parser.parse_args()
    
    import time
    start_time = time.time()
    patchify(args)
    print("patchify cost time {}".format(time.time()-start_time))