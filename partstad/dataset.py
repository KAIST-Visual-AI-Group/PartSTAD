import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import numpy as np
import glob
from tqdm import trange, tqdm
import os
import sys
from sp_utils import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import normalize_pc_o3d


def to_device(data,device):
    
    if isinstance(data,torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
#         import pdb; pdb.set_trace()
        for key in data:
            # if isinstance(data[key],torch.Tensor):
            data[key] = to_device(data[key],device)
        return data
    if isinstance(data,list):
        
        return [to_device(d,device) for d in data]

    return data


class PartSTAD_Dataset(Dataset):
    def __init__(self, data_dir, pre_dir, glip_dir, info_pre_dir, meta, test=False, max_data_per_category=None, category=None, test_list_dir=None):
        
        self.categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","StorageFurniture","Switch","Table","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Chair","Kettle","Lamp","Refrigerator","Suitcase"]

        test_list_dir = test_list_dir if test_list_dir is not None else "val_list.json" # validation list
        with open(test_list_dir,"r") as f:
            test_list = json.load(f)

        if category is not None:
            self.categories = [category]
        
        self.info = []
        for i in range(len(self.categories)):
            category = self.categories[i]
            part_names = meta[category]
            num_label = len(part_names)
            self.pre_dir = pre_dir
            self.glip_dir = glip_dir
            self.info_pre_dir = info_pre_dir
            pc_dirs = glob.glob(f"{data_dir}/{category}/*") # not sorted
            self.num_object = len(pc_dirs)

            if test and max_data_per_category==8:
                pc_dirs = [f"{data_dir}/{category}/{k}" for k in test_list[category]]

            if max_data_per_category is not None:
                num_data = min(len(pc_dirs),max_data_per_category)
            else:
                num_data = len(pc_dirs)
            
            for j in range(num_data):
                pc_dir = pc_dirs[j]
                fname = pc_dir.split("/")[-1]
                fpre_dir = f"{pre_dir}/{category}/{fname}"
                fglip_dir = f"{glip_dir}/{category}/{fname}"
                finfo_pre_dir = f"{info_pre_dir}/{category}/{fname}"
                pc_file = f"{pc_dir}/pc.ply"

                data_dict = {"fname":fname, "category":category, "part_names":part_names, "num_label":num_label, "pc_dir":pc_dir,"fpre_dir":fpre_dir, "fglip_dir":fglip_dir, "finfo_pre_dir":finfo_pre_dir}

                self.info.append(data_dict)

            

        self.train = not test
        if self.train:
            self.init_train_dataset()

        

    
    def init_train_dataset(self):
        print("Load Train Dataset! It could takes some time")
        self.glip_preds = []
        self.glip_features = []
        self.xyzrgbls = []
        self.screen_coords = []
        self.pc_idxs = []
        self.superpoints = []
        self.fnames = []
        self.gt_sp_labels = []
        self.bbox_infos = []
        self.num_sp_for_each_sps = [] # num_sp means number of points for each superpoint
        self.num_sps = []
        self.gt_sp_scores = []
        self.gt_sp_labels = []
        self.valid_sps = []
        self.gt_sem_segs = []

        for i in trange(len(self.info)):
            data_dict = {}
            curr_info = self.info[i]
            fname = curr_info['fname']
            pc_dir = curr_info['pc_dir']
            fpre_dir = curr_info['fpre_dir']
            fglip_dir = curr_info['fglip_dir']
            finfo_pre_dir = curr_info['finfo_pre_dir']
            pc_file = f"{pc_dir}/pc.ply"
            label = np.load(f"{pc_dir}/label.npy", allow_pickle=True)
            label_dict = label.item()
            gt_sem_seg = label_dict['semantic_seg']
            # gt_ins_seg = label_dict['instance_seg']
            data_dict['gt_sem_seg'] = gt_sem_seg

            xyz, rgb = normalize_pc_o3d(pc_file, None, None)
            data_dict['xyzrgbl'] = np.concatenate([xyz,rgb,gt_sem_seg[...,None]],1)

            data_dict['superpoint'] = np.load(f"{fpre_dir}/superpoint.npy", allow_pickle=True)

            with open(f"{fglip_dir}/glip_pred/pred.json", "r") as f:
                glip_pred = json.load(f)
            data_dict['glip_pred'] = glip_pred

            data_dict['glip_feature'] = np.load(f"{fglip_dir}/glip_pred/glip_feature.npz",allow_pickle=True)['feature']

            data_dict['bbox_info'] = np.load(f"{finfo_pre_dir}/bbox_info_compressed.npz", allow_pickle=True)

            try:
                sp_info = np.load(f"{finfo_pre_dir}/sp_info.npz", allow_pickle=True) # TODO
            except:
                sp_info = np.load(f"{fpre_dir}/sp_info.npz", allow_pickle=True)

            data_dict['num_sp'] = sp_info['num_sp']
            data_dict['gt_sp_score'] = sp_info['gt_sp_score']
            data_dict['gt_sp_label'] = sp_info['gt_sp_label']
            data_dict['valid_sp'] = sp_info['valid_sp']

            self.info[i]['data_dict'] = data_dict
        

    def __getitem__(self, idx_):

        if self.train:
            return self._getitem_train(idx_)
        else:
            return self._getitem_test(idx_)


    def _getitem_train(self,idx_):

        info = self.info[idx_]
        fname = info['fname']
        part_names = info['part_names']
        category = info['category']
        data_dict = info['data_dict']
        xyzrgbl = data_dict['xyzrgbl']
        superpoint = data_dict['superpoint']
        glip_pred = data_dict['glip_pred']
        glip_feature = data_dict['glip_feature']
        bbox_info = data_dict['bbox_info']
        sp_map_list = bbox_info['sp_map_list']
        sp_visible_cnt = bbox_info['sp_visible_cnt']
        view_bbox_mask_list = bbox_info['view_bbox_mask_list']
        num_sp = data_dict['num_sp']
        gt_sp_score = data_dict['gt_sp_score']
        gt_sp_label = data_dict['gt_sp_label']
        valid_sp = data_dict['valid_sp']
        gt_sem_seg = data_dict['gt_sem_seg']

        data = {'xyzrgbl':xyzrgbl, 
                     'superpoint':superpoint, 
                     'glip_pred':glip_pred,
                     'glip_feature':glip_feature,
                     'sp_map_list':sp_map_list,
                     'sp_visible_cnt':sp_visible_cnt,
                     'view_bbox_mask_list':view_bbox_mask_list,
                     'num_sp':num_sp,
                     'gt_sp_score':gt_sp_score,
                     'gt_sp_label':gt_sp_label,
                     'valid_sp':valid_sp,
                     'fname':fname,
                     'gt_sem_seg':gt_sem_seg,
                     'part_names':part_names,
                     'category':category
                    }

        return data
    
    def _getitem_test(self,idx_):
        curr_info = self.info[idx_]
        fname = curr_info['fname']
        part_names = curr_info['part_names']
        category = curr_info['category']
        pc_dir = curr_info['pc_dir']
        fpre_dir = curr_info['fpre_dir']
        fglip_dir = curr_info['fglip_dir']
        finfo_pre_dir = curr_info['finfo_pre_dir']
        pc_file = f"{pc_dir}/pc.ply"
        label = np.load(f"{pc_dir}/label.npy", allow_pickle=True)
        label_dict = label.item()
        gt_sem_seg = label_dict['semantic_seg']


        xyz, rgb = normalize_pc_o3d(pc_file, None, None)
        xyzrgbl = np.concatenate([xyz,rgb,gt_sem_seg[...,None]],1)

        superpoint = np.load(f"{fpre_dir}/superpoint.npy", allow_pickle=True)

        with open(f"{fglip_dir}/glip_pred/pred.json", "r") as f:
            glip_pred = json.load(f)

        glip_feature = np.load(f"{fglip_dir}/glip_pred/glip_feature.npz",allow_pickle=True)['feature']

        bbox_info = np.load(f"{finfo_pre_dir}/bbox_info_compressed.npz", allow_pickle=True)
        sp_map_list = bbox_info['sp_map_list']
        sp_visible_cnt = bbox_info['sp_visible_cnt']
        view_bbox_mask_list = bbox_info['view_bbox_mask_list']

        try:
            sp_info = np.load(f"{finfo_pre_dir}/sp_info.npz", allow_pickle=True) # TODO
        except:
            sp_info = np.load(f"{fpre_dir}/sp_info.npz", allow_pickle=True)

        num_sp = sp_info['num_sp']
        gt_sp_score = sp_info['gt_sp_score']
        gt_sp_label = sp_info['gt_sp_label']
        valid_sp = sp_info['valid_sp']

        data = {'xyzrgbl':xyzrgbl, 
                     'superpoint':superpoint, 
                     'glip_pred':glip_pred,
                     'glip_feature':glip_feature,
                     'sp_map_list':sp_map_list,
                     'sp_visible_cnt':sp_visible_cnt,
                     'view_bbox_mask_list':view_bbox_mask_list,
                     'num_sp':num_sp,
                     'gt_sp_score':gt_sp_score,
                     'gt_sp_label':gt_sp_label,
                     'valid_sp':valid_sp,
                     'fname':fname,
                     'gt_sem_seg':gt_sem_seg,
                     'part_names':part_names,
                     'category':category
                    }

        return data

    def __len__(self):
        return len(self.info)

def collate_fn(batch):
    return ([b for b in batch])



# dataset = TrainDataset(data_dir, pre_dir, glip_dir, meta_dir, category)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=n, collate_fn=collate_fn, num_workers = num_workers)
