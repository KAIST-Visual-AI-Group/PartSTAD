import os
import torch
import json
from src.glip_inference import load_model, glip_inference_feat_extract

import argparse
import glob
from tqdm import tqdm, trange

# import open3d as o3d


from PIL import Image

all_categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","Switch","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Kettle","Lamp","Refrigerator","Suitcase", "StorageFurniture", "Table", "Chair"]



def extract_glip_feature(args,meta,category):
    part_names = meta[category]
    num_label = len(part_names)

    config ="GLIP/configs/glip_Swin_L_pt.yaml"
    weight_path = "models/%s.pth" % category
    save_dir = f"{args.glip_dir}/{category}"

    os.makedirs(save_dir, exist_ok=True)

    # Load GLIP model
    print("[loading GLIP model...]")
    glip_model = load_model(config,weight_path)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    pc_dirs = sorted(glob.glob(f"{args.data_dir}/{category}/*"))

    pre_dir = args.preprocess_dir
    img_dir = "rendered_img"

    num_object = len(pc_dirs)
    pbar = trange(num_object)
    for i in pbar:
        pc_dir = pc_dirs[i]
        fname = pc_dir.split("/")[-1]
        fsave_dir = f"{save_dir}/{fname}"
        fpre_dir = f"{pre_dir}/{category}/{fname}"

        num_views = len(glob.glob(f"{fpre_dir}/rendered_img/*.png"))
        num_views = 10

        fimg_dir = f"{fpre_dir}/{img_dir}"
        
        pbar.set_description("GLIP Inference")
        preds = glip_inference_feat_extract(glip_model, fimg_dir, fsave_dir, part_names, save_pred_json=True, not_post_process=False, threshold=args.threshold, num_views=num_views)



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default="Bottle", type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--data_dir', default='data/train', type=str)
    parser.add_argument('--preprocess_dir', default='rendered_pc', type=str)
    parser.add_argument('--glip_dir', default='glip_preprocess/train') #save dir

    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    
    partnete_meta = json.load(open("PartNetE_meta.json"))
    category = args.category
    if category == "all":
        print("Processing For All Categories!")
        for category in tqdm(all_categories[args.start:args.end]):
            print(f"Category : {category}")
            extract_glip_feature(args, partnete_meta, category)
    else:
        extract_glip_feature(args, partnete_meta, category)