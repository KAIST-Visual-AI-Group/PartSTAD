import numpy as np
import os
import argparse
import time
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='Bottle', type=str)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=0, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--ckpt_dir', default='result_training', type=str)
parser.add_argument('--test_dir', default='data/test', type=str)
parser.add_argument('--test_preprocess_dir', default='preprocess/test', type=str)

args = parser.parse_args()


def get_best_ckpt(logfile):
    epoch = 0
    best_epoch = 0
    best_mIoU = -1
    with open(logfile,"r") as f:
        l = f.readlines()
        for line in l:
            lsplit = line.split(" ")
            ind = lsplit[7]
            ind2 = lsplit[8]
            
            if ind[0] == '[':
                epoch = int(lsplit[8][:-1])
                continue
            
            if ind == 'TEST':
                if ind2 == 'RESULT':
                    mIoU = float(lsplit[10].split(":")[-1])

                    if mIoU > best_mIoU:
                        best_mIoU = mIoU
                        best_epoch = epoch
    
    return best_epoch

if __name__ == "__main__":
    ckpt_dir_concat = "_".join(args.ckpt_dir.split("/"))

    category = args.category
    fs = sorted(glob.glob(f"{args.ckpt_dir}/{category}/*"))
    if len(fs) > 0:
        fs=fs[-1]
    else:
        raise Exception("1")

    fs2 = sorted(glob.glob(f'{fs}/ckpt_*.tar'))
    if len(fs2) > 0:
        fs2 = fs2[-1]
    else:
        raise Exception("2")

    best_ckpt = get_best_ckpt(f'{fs}/train.log')
    fs2 = f"{fs}/ckpt_{best_ckpt:03d}.tar"

    script = f'CUDA_VISIBLE_DEVICES={args.device} python run_partstad.py --test --ckpt {fs2} --only_model_weight --eval_category {category} --pos_enc --test_dir {args.test_dir} --test_preprocess_dir {args.test_preprocess_dir}'

    print(script)
    os.system(script)
    time.sleep(5)