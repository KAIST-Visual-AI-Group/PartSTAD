import os
import json
import numpy as np
import argparse
from tqdm import tqdm, trange

from src.rendering_preprocess import rendering_preprocess
from src.extract_glip_feat import extract_glip_feature
from src.sam_pred import sam_predict
from src.weight_pred_preprocess_mask import weight_pred_preprocess

parser = argparse.ArgumentParser()

parser.add_argument('--category', default="Bottle", type=str,
                    help='category name for preprocessing. If category=all, all category will be processed iteratively.') 
parser.add_argument('--start', default=0, type=int, 
                    help='category start index. Only used when the args.category=all') 
parser.add_argument('--end', default=45, type=int, 
                    help='category end index. Only used when the args.category=all')
parser.add_argument('--start2', default=-1, type=int)
parser.add_argument('--end2', default=-1, type=int)
parser.add_argument('--data_dir', default='data/train', type=str, 
                    help='3d data dir')
parser.add_argument('--save_dir', default='preprocess/train', type=str, 
                    help='preprocessed data will be saved here.')

# parser.add_argument('--preprocess_dir', default='preprocess/rendered_pc/train', type=str, 
#                     help='rendered images and meta data will be saved here.')
# parser.add_argument('--glip_dir', default='preprocess/glip_preprocess/train', type=str, 
#                     help='glip predict results will be saved here.') 
# parser.add_argument('--mask_dir', default='preprocess/sam_preprocess/train', type=str, 
#                     help='sam predict results will be saved here.')
# parser.add_argument('--save_dir', default='preprocess/bbox_info_preprocess/train', type=str, 
#                     help='preprocessed data for the weight prediction will be saved here.')

parser.add_argument('--skip_rendering', action='store_true')
parser.add_argument('--skip_glip', action='store_true')
parser.add_argument('--skip_sam', action='store_true')
parser.add_argument('--skip_info_preprocess', action='store_true')

#glip feature extraction option
parser.add_argument('--threshold', default=0.5, type=float)

args = parser.parse_args()

all_categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","Switch","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Kettle","Lamp","Refrigerator","Suitcase", "StorageFurniture", "Table", "Chair"]


def preprocess(args,meta,category):
    save_dir = args.save_dir
    args.preprocess_dir = f"{save_dir}/rendered_pc"
    args.glip_dir = f"{save_dir}/glip_preprocess"
    args.mask_dir = f"{save_dir}/sam_preprocess"
    args.bbox_info_preprocess_dir = f"{save_dir}/bbox_info_preprocess"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.preprocess_dir, exist_ok=True)
    os.makedirs(args.glip_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.bbox_info_preprocess_dir, exist_ok=True)

    if not args.skip_rendering:
        rendering_preprocess(args,meta,category)
    if not args.skip_glip:
        extract_glip_feature(args,meta,category)
    if not args.skip_sam:
        sam_predict(args, meta, category)
    if not args.skip_info_preprocess:    
        weight_pred_preprocess(args,meta,category)

if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json"))
    category = args.category
    if category == "all":
        print("Rendering For All Categories!")
        for category in tqdm(all_categories[args.start:args.end]):
            print(f"Category : {category}")
            preprocess(args, partnete_meta, category)
    else:
        preprocess(args, partnete_meta, category)