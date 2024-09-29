import os
import json
import numpy as np
import argparse
from tqdm import tqdm, trange
import glob
import time
from src.ply2gif import load_and_normalize_point_cloud, create_rotation_gif

base_dirs = sorted(glob.glob(f"demo_examples/*"))
for base_dir in base_dirs:
    category = base_dir.split("/")[-1]
    print(f"Run Preprocessing - {category}")
    os.system(f"python preprocess.py --category {category} --data_dir demo_examples --save_dir demo_examples_result/preprocess")
    time.sleep(1)

    ckpt = f"demo_examples/ckpts/{category}_ckpt.tar"
    print(f"Run Evaluation - {category}")
    os.system(f"python run_partstad.py --test --ckpt {ckpt} --only_model_weight --category {category} --test_dir demo_examples --test_preprocess_dir demo_examples_result/preprocess --eval_save_dir demo_examples_result/test_result --visualize_segment")
    time.sleep(1)

    print(f"Run Visualization - {category}")
    os.makedirs("demo_examples_result/visualization", exist_ok=True)
    ply_file_paths = sorted(glob.glob(f"demo_examples_result/test_result/{category}/*/semantic_seg/all.ply"))
    for ply_file_path in ply_file_paths:
        fid = ply_file_path.split("/")[-3]
        points, colors = load_and_normalize_point_cloud(ply_file_path)
        create_rotation_gif(points, colors, gif_filename=f"demo_examples_result/visualization/{category}_{fid}.gif")

