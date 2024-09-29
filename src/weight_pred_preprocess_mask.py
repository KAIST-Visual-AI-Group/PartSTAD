import os
import torch
import json
import numpy as np
from src.utils import normalize_pc_o3d
from src.bbox2seg import get_visible_point_mask, get_sp_visible_cnt
from src.sp_utils import get_sp_label, get_sp_score

import argparse
import glob
from tqdm import tqdm, trange

import imageio as io
import time

# import open3d as o3d
from PIL import Image
import pickle

all_categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","Switch","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Kettle","Lamp","Refrigerator","Suitcase", "StorageFurniture", "Table", "Chair"]

def weight_pred_preprocess(args,meta,category):
    part_names = meta[category]
    num_label = len(part_names)

    save_dir = f"{args.bbox_info_preprocess_dir}/{category}"

    pre_dir = args.preprocess_dir

    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    pc_dirs = sorted(glob.glob(f"{args.data_dir}/{category}/*"))

    meta_dir = "img_meta"

    num_object = len(pc_dirs)
    num_views = 10 #TODO: to be changed


    glip_dir = args.glip_dir
    mask_dir = args.mask_dir


    if (args.start2 != -1) and (args.end2 != -1):
        tr = trange(args.start2, min(args.end2, num_object))
    else:
        tr = trange(num_object)

    for i in tr:
        tr.set_description("BBox info preprocessing for weight prediction")
        
        pc_dir = pc_dirs[i]
        fname = pc_dir.split("/")[-1]
        fsave_dir = f"{save_dir}/{fname}"
        os.makedirs(fsave_dir, exist_ok=True)
        fpre_dir = f"{pre_dir}/{category}/{fname}"
        fglip_dir = f"{glip_dir}/{category}/{fname}"
        fmask_dir = f"{mask_dir}/{category}/{fname}"
        pc_file = f"{pc_dir}/pc.ply"
        label = np.load(f"{pc_dir}/label.npy", allow_pickle=True)
        label_dict = label.item()
        gt_sem_seg = label_dict['semantic_seg']

        xyz, rgb = normalize_pc_o3d(pc_file, fsave_dir, device)

        screen_coords = np.load(f"{fpre_dir}/{meta_dir}/screen_coords.npy", allow_pickle=True)
        pc_idx = np.load(f"{fpre_dir}/{meta_dir}/pc_idx.npy", allow_pickle=True)
        superpoint = np.load(f"{fpre_dir}/superpoint.npy", allow_pickle=True)

        with open(f"{fglip_dir}/glip_pred/pred.json", "r") as f:
            glip_preds = json.load(f)

        masks = np.load(f"{fmask_dir}/sam_mask.npz")['mask']

        sp_visible_cnt_per_view = get_sp_visible_cnt(xyz, superpoint, glip_preds, screen_coords, pc_idx, part_names, device, num_views)
        np.savez_compressed(f"{fsave_dir}/sp_visible_cnt_per_view.npz", sp_visible_cnt_per_view=sp_visible_cnt_per_view)

        # get GT score for each superpoint. 
        gt_sp_score, num_sp = get_sp_score(superpoint, gt_sem_seg, num_label)
        # get GT label for each superpoint.
        gt_sp_label = get_sp_label(superpoint, gt_sem_seg) # -1 for the null label, 0,1,~ for part labels


        sp_map_list, sp_visible_cnt, view_bbox_mask_list, valid_bbox_idx_list = get_visible_point_mask(xyz, superpoint, glip_preds, masks, screen_coords, pc_idx, part_names, device, num_view=num_views)


        valid_sp = np.ones((len(superpoint)))
        for k in range(len(superpoint)):
            if np.sum(sp_map_list[k]) == 0:
                valid_sp[k] = 0

        visible_sp = sp_visible_cnt != 0
        valid_sp_visible_sp = valid_sp*visible_sp


        np.savez_compressed(f"{fsave_dir}/sp_info.npz", gt_sp_score=gt_sp_score, num_sp=num_sp, gt_sp_label = gt_sp_label, valid_sp=valid_sp)

        sp_map_list = np.asarray(sp_map_list, dtype='object')
        np.savez_compressed(f"{fsave_dir}/bbox_info_compressed.npz", sp_map_list=sp_map_list,sp_visible_cnt=sp_visible_cnt,view_bbox_mask_list=view_bbox_mask_list,valid_bbox_idx_list=valid_bbox_idx_list)

        # try:
        #     np.savez_compressed(f"{fsave_dir}/bbox_info_compressed.npz", sp_map_list=sp_map_list,sp_visible_cnt=sp_visible_cnt,view_bbox_mask_list=view_bbox_mask_list,valid_bbox_idx_list=valid_bbox_idx_list)
        # except:
        #     data_dict = {'sp_map_list':sp_map_list,'sp_visible_cnt':sp_visible_cnt,'view_bbox_mask_list':view_bbox_mask_list,'valid_bbox_idx_list':valid_bbox_idx_list}
        #     with open(f"{fsave_dir}/bbox_info_compressed.npz", 'wb') as f:
        #         pickle.dump(data_dict, f, protocol=4)

    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default="Bottle", type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--data_dir', default='data/train', type=str)
    parser.add_argument('--preprocess_dir', default='rendered_pc', type=str)
    parser.add_argument('--bbox_info_preprocess_dir', default='bbox_info_preprocess')
    parser.add_argument('--glip_dir', default='glip_preprocess/train', type=str)
    parser.add_argument('--mask_dir', default='sam_preprocess')

    parser.add_argument('--start2', default=-1, type=int)
    parser.add_argument('--end2', default=-1, type=int)

    args = parser.parse_args()

    partnete_meta = json.load(open("PartNetE_meta.json"))
    category = args.category
    if category == "all":
        print("Rendering For All Categories!")
        for category in tqdm(all_categories[args.start:args.end]):
            print(f"Category : {category}")
            weight_pred_preprocess(args, partnete_meta, category)
    else:
        weight_pred_preprocess(args, partnete_meta, category)