import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc_o3d
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint

import argparse
import glob
from tqdm import tqdm, trange

# import requests
from PIL import Image
import time
import matplotlib.pyplot as plt
# from diffusers import StableDiffusionDepth2ImgPipeline

cmap = plt.get_cmap("turbo")

all_categories = [
    "Bottle","Box","Bucket","Camera","Cart",
    "Clock","CoffeeMachine","Dishwasher","Dispenser","Display",
    "Door","Eyeglasses","Faucet","FoldingChair","Globe",
    "Keyboard","KitchenPot","Knife","Laptop","Lighter",
    "Microwave","Mouse","Oven","Pen","Phone",
    "Pliers","Printer","Remote","Safe","Scissors",
    "Stapler","Switch","Toaster","Toilet","TrashCan",
    "USB","WashingMachine","Window","Kettle","Lamp",
    "Refrigerator","Suitcase","StorageFurniture","Table","Chair"]

print(f"Num Categories: {len(all_categories)}")

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def get_seg_color(seg, num_label=None):
    num_pts = seg.shape[0]
    segs = np.unique(seg)
    num_segs = segs.shape[0] if num_label is None else num_label
    rgb = np.zeros(num_pts)
    for i in range(num_segs):
        val = (i+1)/(num_segs+1)
        idx = segs[i] if num_label is None else i
        if idx == -1:
            val = 0
        rgb[seg==idx] = val
    
    rgb = cmap(rgb)[:,:3]
    return rgb



def rendering_preprocess(args,meta,category):
    part_names = meta[category]
    num_label = len(part_names)

    print(f"num label : {num_label}")
    print(f"part names : {part_names}")

    
    save_dir = args.preprocess_dir

    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()

    pc_dirs = sorted(glob.glob(f"{args.data_dir}/{category}/*"))

    num_object = len(pc_dirs)
    if (args.start2 != -1) and (args.end2 != -1):
        r = range(args.start2, min(args.end2, num_object))
    else:
        r = range(num_object)

    pbar = tqdm(r)
    for i in pbar:
        pc_dir = pc_dirs[i]
        fname = pc_dir.split("/")[-1]
        fsave_dir = f"{save_dir}/{category}/{fname}"
        pc_file = f"{pc_dir}/pc.ply"
        label = np.load(f"{pc_dir}/label.npy", allow_pickle=True)
        label_dict = label.item()
        gt_sem_seg = label_dict['semantic_seg']
        gt_ins_seg = label_dict['instance_seg']
        seg_color = get_seg_color(gt_sem_seg,num_label=num_label)
        ins_color = get_seg_color(gt_ins_seg)
        img_meta_dir = f"{fsave_dir}/img_meta"
        os.makedirs(img_meta_dir, exist_ok=True)


        xyz, rgb = normalize_pc_o3d(pc_file, fsave_dir, io, device)
        
        pbar.set_description("Rendering Pointcloud")
        img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, fsave_dir, device)
        np.save(f"{img_meta_dir}/pc_idx.npy", pc_idx)
        np.save(f"{img_meta_dir}/screen_coords.npy", screen_coords)
        img_dir, pc_idx, screen_coords = render_pc(xyz, seg_color, fsave_dir, device, dir_name="rendered_sem_seg")
        img_dir, pc_idx, screen_coords = render_pc(xyz, ins_color, fsave_dir, device, dir_name="rendered_ins_seg")
        
        # Generate Superpoint / It takes the longest time!
        pbar.set_description("Generating Superpoint! It takes the longest time :(")
        superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=fsave_dir, save=True)
        np.save(f"{fsave_dir}/superpoint.npy", superpoint)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default="Bottle", type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--data_dir', default='data/train', type=str)
    parser.add_argument('--preprocess_dir', default='rendered_pc') #save dir
    parser.add_argument('--start2', default=-1, type=int)
    parser.add_argument('--end2', default=-1, type=int)

    args = parser.parse_args()

    partnete_meta = json.load(open("PartNetE_meta.json")) 
    category = args.category
    if category == "all":
        print("Rendering For All Categories!")
        for category in tqdm(all_categories[args.start:args.end]):
            print(f"Category : {category}")
            rendering_preprocess(args, partnete_meta, category)
    else:
        rendering_preprocess(args, partnete_meta, category)
        