import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import sys
from datetime import datetime
import logging as log

from src.bbox2seg import score2seg, score2seg_without_threshold
from src.sp_utils import *


import argparse
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
cmap = plt.get_cmap('turbo')

from partstad.models import WeightPredNetworkCNe
from partstad.dataset import PartSTAD_Dataset, to_device, collate_fn



import random
import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

#Directory
parser.add_argument('--test_dir', default='data/test', type=str)
parser.add_argument('--train_dir', default='data/train', type=str) #train data dir
parser.add_argument('--train_preprocess_dir', default='preprocess/train', type=str) #train preprocess data dir
parser.add_argument('--test_preprocess_dir', default='preprocess/test', type=str) #test preprocess data dir

parser.add_argument('--save_dir', default='result_training')


# Dataset options
parser.add_argument('--max_data_per_category',default=999, type=int)
parser.add_argument('--max_testdata_per_category',default=8, type=int)
# parser.add_argument('--train_category', default="", type=str) #for eval
# parser.add_argument('--eval_category', default="", type=str) #for eval
parser.add_argument('--category', default="", type=str)
parser.add_argument('--start', default=0, type=int) #for eval
parser.add_argument('--end', default=0, type=int) #for eval


# Mode selection
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')


#Training options
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--init_weight', default=10.0, type=float)
parser.add_argument('--save_init', action='store_true')
parser.add_argument('--epoch', default=400, type=int)
parser.add_argument('--model_lr', default=0.0001, type=float)
parser.add_argument('--param_lr', default=0.05, type=float)
parser.add_argument('--riou_loss_weight', default=1.0, type=float)
parser.add_argument('--ce_loss_weight', default=0.0, type=float)
parser.add_argument('--save_weight', action='store_true')
parser.add_argument('--lr_decay_rate_network', default=1, type=float)
parser.add_argument('--lr_decay_rate_weight', default=1, type=float)
parser.add_argument('--network_training_start_epoch', default=0, type=int) # network training starts from network_start_epoch. 
parser.add_argument('--network_training_end_epoch', default=10000, type=int) # network training ends from network_start_epoch. 
parser.add_argument('--null_label_score_training_start_epoch', default=0, type=int) # null_label_score training starts from null_label_score_start_epoch
parser.add_argument('--null_label_score_training_end_epoch', default=100000, type=int) # null_label_score training ends from null_label_score_start_epoch


# Validation options
parser.add_argument('--test_intv', default=1, type=int)
parser.add_argument('--save_intv', default=1, type=int)


# Test options
parser.add_argument('--ckpt', default="", type=str) #load ckpt (including optimizer), for eval
parser.add_argument('--only_model_weight', action='store_true') # only load model weight, not load optimizer.
parser.add_argument('--best', action='store_true') # evaluate for best ckpt(evaluation option) (based on the mean of part mIoU)
parser.add_argument('--visualize_segment', action='store_true')


# Model options
parser.add_argument('--network', default='linear_cn', type=str, choices=['linear_cn']) #cn : context normalization
parser.add_argument('--num_block', default=0, type=int)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--num_cn_layer', default=1, type=int)
parser.add_argument('--no_skip_connection', action='store_true')
parser.add_argument('--zero_init', action='store_true')
parser.add_argument('--he_init', action='store_true')
parser.add_argument('--no_pos_enc', action='store_true') #True

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)


all_categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","StorageFurniture","Switch","Table","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Chair","Kettle","Lamp","Refrigerator","Suitcase"]


print(len(all_categories))

# Positional Encoding Code from NeRF pytorch
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 7,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class LossReducer(object):
    def __init__(self,loss_weight_dict):
        self.loss_weights = loss_weight_dict
    
    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k] 
        for k in self.loss_weights.keys()])
        
        return total_loss

class PartSTAD():
    def __init__(self,meta,args,eval=False, category=None):

        self.meta = meta

        # dirs
        self.args = args
        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        self.save_dir = args.save_dir

        self.pre_dir = f"{args.train_preprocess_dir}/rendered_pc"
        self.glip_dir = f"{args.train_preprocess_dir}/glip_pred"
        self.info_pre_dir = f"{args.train_preprocess_dir}/bbox_info_preprocess"

        self.test_pre_dir = f"{args.test_preprocess_dir}/rendered_pc"
        self.test_glip_dir = f"{args.test_preprocess_dir}/glip_pred"
        self.test_info_pre_dir = f"{args.test_preprocess_dir}/bbox_info_preprocess"

        self.test_intv = args.test_intv
        self.save_intv = args.save_intv

        self.visualize_segment = False # args.visualize_segment only work for eval

        
        # define device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            raise Exception("GPU is needed!")

        if not eval:
            self.train_dataset = PartSTAD_Dataset(self.train_dir, self.pre_dir, self.glip_dir, self.info_pre_dir, meta, max_data_per_category=args.max_data_per_category, category=category)

            self.test_dataset = PartSTAD_Dataset(self.test_dir, self.test_pre_dir, self.test_glip_dir, self.test_info_pre_dir, meta, test=True, max_data_per_category=args.max_testdata_per_category, category=category)


        self.batch_size = args.batch_size

        self.num_views = 10 # TODO

        self.start_epoch = 1
        self.epoch = args.epoch

        self.network_training_start_epoch = args.network_training_start_epoch
        self.network_training_end_epoch = args.network_training_end_epoch
        self.null_label_score_training_start_epoch = args.null_label_score_training_start_epoch
        self.null_label_score_training_end_epoch = args.null_label_score_training_end_epoch

        self.update_null_weight = True

        self.skip_connection = not self.args.no_skip_connection

        self.init_weight = args.init_weight

        self.in_channel = 256

        self.pos_enc =  not args.no_pos_enc # default: True
        if self.pos_enc:
            embed_fn, embed_dim = get_embedder(10)
            self.embed_fn = embed_fn
            self.embed_dim = embed_dim
            self.in_channel = self.in_channel + embed_dim
            self.view_dir = np.load("view.npy",allow_pickle=True) # 10,3 vector which represent view direction

        self.weight_pred_network = self.get_network()
        self.weight_pred_network = self.weight_pred_network.to(self.device)

        self.null_label_weight = torch.nn.Parameter(torch.ones(1,device=self.device)*args.init_weight)

        self.lr_decay_rate_network = args.lr_decay_rate_network
        self.lr_decay_rate_weight = args.lr_decay_rate_weight

        self.optimizer_network = torch.optim.AdamW(
            params=[{'params': self.weight_pred_network.parameters(), 'lr': args.model_lr}],
            betas=(0.9, 0.999),
         )
        gamma_network = pow(self.lr_decay_rate_network,1/self.epoch)
        self.scheduler_network = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_network, lr_lambda=lambda epoch: gamma_network ** epoch)

        self.optimizer_weight = torch.optim.AdamW(
            params=[{'params': [self.null_label_weight], 'lr': args.param_lr}],
            betas=(0.9, 0.999),
         )
        gamma_weight = pow(self.lr_decay_rate_weight,1/self.epoch)
        self.scheduler_weight = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_weight, lr_lambda=lambda epoch: gamma_weight ** epoch)

        if len(args.ckpt) > 0:
            self.load_ckpt(args.ckpt, args.only_model_weight)
            self.ckpt = args.ckpt


        loss_weight_dict = {'riou_loss':args.riou_loss_weight, 'ce_loss':args.ce_loss_weight}
        self.loss_reducer = LossReducer(loss_weight_dict)
        self.entropy_loss = torch.nn.CrossEntropyLoss()

        if not eval:
            self.create_dir()
            # self.load_checkpoint()
            self.logger = log.getLogger()
            self.logger.setLevel(log.INFO)
            formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler = log.FileHandler(f'{self.save_dir}/train.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_network(self):

        if self.args.network == 'linear_cn':
            return WeightPredNetworkCNe(in_channel=self.in_channel,num_cn_layer=self.args.num_cn_layer, he_init=self.args.he_init, skip_connection=self.skip_connection)
        else:
            raise Exception('Network should be one of [linear_cn]')


    def load_ckpt(self, ckpt, only_model_weight=False):
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.weight_pred_network.state_dict(),
            'optimizer_network_state_dict': self.optimizer_network.state_dict(),
            'optimizer_weight_state_dict': self.optimizer_weight.state_dict(),
            'null_label_weight': self.null_label_weight.data.item(),
        }, f"{self.save_dir}/ckpt_{epoch:03d}.tar")
        """

        ckpt_dict = torch.load(ckpt)
        self.weight_pred_network.load_state_dict(ckpt_dict['model_state_dict'])
        self.null_label_weight.data[0] = ckpt_dict['null_label_weight']

        if not only_model_weight:
            self.optimizer_network.load_state_dict(ckpt_dict['optimizer_network_state_dict'])
            self.optimizer_weight.load_state_dict(ckpt_dict['optimizer_weight_state_dict'])
            self.start_epoch = ckpt_dict['epoch']
        print(f"load data from checkpoint file : {ckpt}")


    def train(self):

        print(f"Start Training!")
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers = 0, shuffle=True)
        len_dataloader = len(train_dataloader)
        print("len_dataloader", len_dataloader)
        pbar = trange(self.start_epoch,self.epoch+1)
        epoch_loss = 0

        iou_dict = {}
        test_iou_dict = {}

        total_loss_save = np.zeros(self.epoch*len_dataloader)
        avg_loss_save = np.zeros(self.epoch)
        test_loss_save = np.zeros(self.epoch//self.test_intv)

        self.weight_pred_network.train()
        for epoch in pbar:
            avg_loss = 0
            extra_dict_merged = {}

            # Training Iteration
            for it, batch in enumerate(train_dataloader):
                self.optimizer_network.zero_grad()
                self.optimizer_weight.zero_grad()
                loss_dict, extra_dict = self.train_one_step(batch)

                total_loss = self.loss_reducer(loss_dict)
                total_loss_save[(epoch-1)*len_dataloader+it] = total_loss.item() #iter loss
                extra_dict_merged.update(extra_dict)

                total_loss.backward()

                if (epoch >= self.network_training_start_epoch) and (epoch <= self.network_training_end_epoch):
                    self.optimizer_network.step()
                    
                if (epoch >= self.null_label_score_training_start_epoch) and (epoch <= self.null_label_score_training_end_epoch):
                    self.optimizer_weight.step()

                avg_loss += total_loss.item()/len_dataloader 
                loss_description = f"[Epoch {epoch:03d}] AVG loss:{round(epoch_loss,5)}|Iter {it}/{len_dataloader} loss:{round(total_loss.item(),5)}|Null_w:{round(self.null_label_weight.data.item(),5)}" 
                pbar.set_description(loss_description)
                self.logger.info(loss_description)

            self.scheduler_network.step()
            self.scheduler_weight.step()
            
            avg_loss_save[epoch-1] = avg_loss # epoch loss

            # Logging
            epoch_loss = (epoch_loss*(epoch-1) + avg_loss)/(epoch) # total average loss

            iou_dict, mean_miou, mean_miou_without_threshold = self.log_iou(extra_dict_merged, epoch, iou_dict)
            self.logger.info(f"TRAIN RESULT | mIoU:{round(mean_miou,5)} | mIoU w/o threshold:{round(mean_miou_without_threshold,5)} | NULL weight:{round(self.null_label_weight.data.item(),5)} | network LR:{round(self.optimizer_network.param_groups[0]['lr'],9)}")


            if (epoch % self.save_intv == 0) and (epoch != 0):

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.weight_pred_network.state_dict(),
                    'optimizer_network_state_dict': self.optimizer_network.state_dict(),
                    'optimizer_weight_state_dict': self.optimizer_weight.state_dict(),
                    'null_label_weight': self.null_label_weight.data.item(),
                }, f"{self.save_dir}/ckpt_{epoch:03d}.tar")

            if epoch % self.test_intv == 0:
                self.weight_pred_network.eval()
                with torch.no_grad():
                    test_loss_dict, test_extra_dict, test_iou_dict = self.test(iou_dict=test_iou_dict,epoch=epoch)
                    test_loss_save[epoch//self.test_intv-1] = self.loss_reducer(test_loss_dict).item()
                self.weight_pred_network.train()
            
        print("Training End!")
        #log whole ious
        print("Start logging result")
        self.plot_iou_dict(iou_dict)
        self.plot_iou_dict(test_iou_dict, intv=self.test_intv, plot_dir="test_plots", iou_dir="test_iou_txt", test=True)
        np.save(f"{self.save_dir}/total_loss_save.npy", total_loss_save)
        np.save(f"{self.save_dir}/avg_loss_save.npy", avg_loss_save)
        np.save(f"{self.save_dir}/test_loss_save.npy", test_loss_save)

        #After training ends, starts test
        self.weight_pred_network.eval()
        with torch.no_grad():
            self.test()
            print("Testing End!")

    def train_one_step(self, batch):
        
        loss_dict = {
            'riou_loss':0,
            'ce_loss':0,
        }

        extra_dict = {}

        len_batch = len(batch)
        
        for data in batch: # cannot control multiple data at ones
            
            category = data['category']
            part_names = data['part_names']
            num_label = len(part_names)

            xyzrgbl = data['xyzrgbl']
            xyz = xyzrgbl[:,:3]
            gt_sem_seg = xyzrgbl[:,-1]
            superpoint = data['superpoint']

            glip_pred = data['glip_pred']
            glip_feature = data['glip_feature']

            sp_map_list = data['sp_map_list']
            sp_visible_cnt = data['sp_visible_cnt']
            view_bbox_mask_list = data['view_bbox_mask_list']

            num_sp = data['num_sp']
            gt_sp_score = data['gt_sp_score']
            gt_sp_label = data['gt_sp_label']

            valid_sp = data['valid_sp']

            #to device
            gt_sp_score_torch, num_sp_torch = torch.from_numpy(gt_sp_score).to(self.device), torch.from_numpy(num_sp).to(self.device)

            gt_sp_label_torch = torch.from_numpy(gt_sp_label).type(torch.long).to(self.device)

            valid_sp = torch.from_numpy(valid_sp).type(torch.bool).to(self.device)

            sp_map_list = [torch.from_numpy(sp_map_list[k]).to(self.device) for k in range(len(sp_map_list))]
            view_bbox_mask_list = [torch.from_numpy(view_bbox_mask_list[k]).to(self.device) for k in range(len(sp_map_list))] 
            sp_visible_cnt = torch.from_numpy(sp_visible_cnt).to(self.device)

            visible_sp = sp_visible_cnt != 0
            valid_sp = valid_sp*visible_sp

            glip_feature = torch.from_numpy(glip_feature).to(self.device)

            if self.pos_enc:
                pos_vecs = []
                for pred in glip_pred:
                    id = pred['image_id']
                    x,y,h,w = pred['bbox']
                    vx,vy,vz = self.view_dir[id]
                    x1,y1,x2,y2 = x/800,y/800,(x+w)/800,(y+h)/800
                    pos_vec = np.array([vx,vy,vz,x1,y1,x2,y2])
                    pos_vecs.append(pos_vec)
                pos_vecs = np.stack(pos_vecs,0) # num_pred, 7
                pos_vecs_torch = torch.from_numpy(pos_vecs).type(torch.FloatTensor).to(self.device)
                pos_embed = self.embed_fn(pos_vecs_torch)
                glip_feature = torch.cat([glip_feature, pos_embed],1)

            bbox_weight = F.relu(self.init_weight + self.weight_pred_network(glip_feature))
            null_label_weight = F.relu(self.null_label_weight)

            whole_weight = torch.cat([null_label_weight, bbox_weight.squeeze()])
            # print(f"whole weight : {whole_weight}")

            sem_score = compute_score(superpoint, num_label, self.num_views, self.device, sp_visible_cnt, sp_map_list, whole_weight, view_bbox_mask_list, valid_sp, update_null_weight=self.update_null_weight)

            sem_score = F.softmax(sem_score, dim=1)

            riou, _ = compute_riou(gt_sp_score_torch[valid_sp], sem_score[valid_sp], num_sp_torch[valid_sp])
            riou_loss = torch.mean(1.-riou[1:]) # exclude null label riou
            loss_dict['riou_loss'] += riou_loss/len_batch

            ce_loss = self.entropy_loss(sem_score[valid_sp], gt_sp_label_torch[valid_sp]+1)
            loss_dict['ce_loss'] += ce_loss/len_batch


            if True: # TODO: change to self.args.compute_iou:
                fname = data['fname']
                gt_sem_seg = data['gt_sem_seg']
                fsave_dir = f"{self.save_dir}/{fname}"
                sem_seg, _ = score2seg(xyz,sem_score.cpu().detach().numpy(),superpoint,part_names,fsave_dir, visualize=False, valid_sp=valid_sp)
                seg_ious, seg_miou, seg_count = compute_iou(gt_sem_seg,sem_seg,num_label)

                sem_seg_without_threshold, _ = score2seg_without_threshold(xyz,sem_score.cpu().detach().numpy(),superpoint,part_names,fsave_dir, visualize=False, valid_sp=valid_sp)
                seg_ious_without_threshold, seg_miou_without_threshold, seg_count_without_threshold = compute_iou(gt_sem_seg,sem_seg_without_threshold,num_label)

                extra_dict[f"{category}_{fname}"] = {"ious":seg_ious, "count":seg_count, "rious":riou.clone().detach().cpu().numpy(), "ious_without_threshold":seg_ious_without_threshold, "miou":seg_miou, "miou_without_threshold":seg_miou_without_threshold}

        return loss_dict, extra_dict


    def test(self, iou_dict=None, epoch=None):

        print("Start testing!")

        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, collate_fn=collate_fn, num_workers = 0, shuffle=False)
        len_dataloader = len(test_dataloader)
        print("len test_dataloader : ",len_dataloader)

        if iou_dict is None:
            iou_dict = {}
        if epoch is None:
            fn = "test_result"
            cnt = 0
        else:
            fn = f"test_result_{epoch:03d}"
            cnt = epoch
        
        extra_dict = {}
        loss_dict = {'riou_loss':0, 'ce_loss':0}
        pbar = tqdm(enumerate(test_dataloader))
        for it, batch in pbar:
            pbar.set_description(f"iter:{it+1}/{len_dataloader}")
            for data in batch: # cannot control multiple data at ones
                category = data['category']
                part_names = data['part_names']
                num_label = len(part_names)

                xyzrgbl = data['xyzrgbl']
                xyz = xyzrgbl[:,:3]
                gt_sem_seg = xyzrgbl[:,-1]
                superpoint = data['superpoint']

                glip_pred = data['glip_pred']
                glip_feature = data['glip_feature']

                sp_map_list = data['sp_map_list']
                sp_visible_cnt = data['sp_visible_cnt']
                view_bbox_mask_list = data['view_bbox_mask_list']

                num_sp = data['num_sp']
                gt_sp_score = data['gt_sp_score']
                gt_sp_label = data['gt_sp_label']

                valid_sp = data['valid_sp']

                #to device
                gt_sp_score_torch, num_sp_torch = torch.from_numpy(gt_sp_score).to(self.device), torch.from_numpy(num_sp).to(self.device)

                gt_sp_label_torch = torch.from_numpy(gt_sp_label).type(torch.long).to(self.device)

                valid_sp = torch.from_numpy(valid_sp).type(torch.bool).to(self.device)

                sp_map_list = [torch.from_numpy(sp_map_list[k]).to(self.device) for k in range(len(sp_map_list))]
                view_bbox_mask_list = [torch.from_numpy(view_bbox_mask_list[k]).to(self.device) for k in range(len(sp_map_list))] 
                sp_visible_cnt = torch.from_numpy(sp_visible_cnt).to(self.device)

                visible_sp = sp_visible_cnt != 0
                valid_sp = valid_sp*visible_sp

                glip_feature = torch.from_numpy(glip_feature).to(self.device)

                if self.pos_enc:
                    pos_vecs = []
                    for pred in glip_pred:
                        id = pred['image_id']
                        x,y,h,w = pred['bbox']
                        vx,vy,vz = self.view_dir[id]
                        x1,y1,x2,y2 = x/800,y/800,(x+w)/800,(y+h)/800
                        pos_vec = np.array([vx,vy,vz,x1,y1,x2,y2])
                        pos_vecs.append(pos_vec)
                    pos_vecs = np.stack(pos_vecs,0) # num_pred, 7
                    pos_vecs_torch = torch.from_numpy(pos_vecs).type(torch.FloatTensor).to(self.device)
                    pos_embed = self.embed_fn(pos_vecs_torch)
                    glip_feature = torch.cat([glip_feature, pos_embed],1)

                bbox_weight = F.relu(self.init_weight + self.weight_pred_network(glip_feature))
                null_label_weight = F.relu(self.null_label_weight)
                whole_weight = torch.cat([null_label_weight, bbox_weight.squeeze()])

                sem_score = compute_score(superpoint, num_label, self.num_views, self.device, sp_visible_cnt, sp_map_list, whole_weight, view_bbox_mask_list, valid_sp, update_null_weight=self.update_null_weight)

                sem_score = F.softmax(sem_score, dim=1)

                riou, _ = compute_riou(gt_sp_score_torch[valid_sp], sem_score[valid_sp], num_sp_torch[valid_sp])
                riou_loss = torch.mean(1.-riou[1:]) # exclude null label riou
                ce_loss = self.entropy_loss(sem_score[valid_sp], gt_sp_label_torch[valid_sp]+1)

                loss_dict['riou_loss'] += riou_loss
                loss_dict['ce_loss'] += ce_loss

                fname = data['fname']
                gt_sem_seg = data['gt_sem_seg']
                fsave_dir = f"{self.save_dir}/{fname}"
                if self.visualize_segment:
                    os.makedirs(fsave_dir, exist_ok=True)
                
                # if self.save_weight:
                #     for pred in glip_pred:
                #         pred['weight'] 
                    
                #     glip_save(pred, f"{self.test_pre_dir}/{category}", )
                # sem_seg, _ = score2seg(xyz,sem_score.cpu().detach().numpy(),superpoint,part_names,fsave_dir, visualize=False, valid_sp=valid_sp)
                sem_seg, _ = score2seg(xyz,sem_score.cpu().detach().numpy(),superpoint,part_names,fsave_dir, visualize=self.visualize_segment, valid_sp=valid_sp)
                seg_ious, seg_miou, seg_count = compute_iou(gt_sem_seg,sem_seg,num_label)

                sem_seg_without_threshold, _ = score2seg_without_threshold(xyz,sem_score.cpu().detach().numpy(),superpoint,part_names,fsave_dir, visualize=False, valid_sp=valid_sp)
                seg_ious_without_threshold, seg_miou_without_threshold, _ = compute_iou(gt_sem_seg,sem_seg_without_threshold,num_label)

                self.logger.info(f"TEST {it+1}/{len_dataloader} Id:{fname} | mIoU:{round(seg_miou,5)} | mIoU w/o threshold:{round(seg_miou_without_threshold,5)}")

                extra_dict[f"{category}_{fname}"] = {"ious":seg_ious, "count":seg_count, "rious":riou.clone().detach().cpu().numpy(), "ious_without_threshold":seg_ious_without_threshold, "miou":seg_miou, "miou_without_threshold":seg_miou_without_threshold}

        iou_dict, mean_miou, mean_miou_without_threshold = self.log_iou(extra_dict, cnt, iou_dict, intv=self.test_intv)
        self.logger.info(f"TEST RESULT | mIoU:{round(mean_miou,5)} | mIoU w/o threshold:{round(mean_miou_without_threshold,5)} | NULL weight:{round(self.null_label_weight.data.item(),5)}")

        return loss_dict, extra_dict, iou_dict


    def eval(self, category=None):
        try:
            ckpt_dir = self.ckpt
            if self.args.best:
                ckpt_fname = 'best'
            else:
                ckpt_fname = ckpt_dir.split("/")[-1][:-4]
            self.save_dir = "/".join(ckpt_dir.split("/")[:-1]) + f"/eval_{ckpt_fname}/{category}"
            os.makedirs(self.save_dir, exist_ok=True)
        except:
            raise Exception(f"For evaluation, ckpt, category should not be None.")

        self.visualize_segment = self.args.visualize_segment 
        self.save_weight = self.args.save_weight
        
        self.logger = log.getLogger()
        self.logger.setLevel(log.INFO)
        formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = log.FileHandler(f'{self.save_dir}/eval.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        print(f"Start evaluation for {category}")
        self.test_intv = 1
        self.epoch = 1
        print(f"test dir: {self.test_dir}")
        print(f"test glip dir: {self.test_glip_dir}")
        
        self.test_dataset = PartSTAD_Dataset(self.test_dir, self.test_pre_dir, self.test_glip_dir, self.test_info_pre_dir, self.meta, test=True, max_data_per_category=9999, category=category)

        self.weight_pred_network.eval()
        with torch.no_grad():
            loss_dict, extra_dict, iou_dict = self.test()
        self.save_iou_eval(extra_dict, category)
    
    def save_iou_eval(self, extra_dict, category):
        total_seg_iou = None
        total_seg_iou_wo_threshold = None
        total_seg_count = None
        part_names = self.meta[category]
        os.makedirs(f"{self.save_dir}/ious", exist_ok=True)
        os.makedirs(f"{self.save_dir}/ious_wo_threshold", exist_ok=True)
        for k,v in extra_dict.items():
            fname = k
            iou = v['ious']
            iou_wo_threshold = v['ious_without_threshold']
            count = v['count']

            if total_seg_iou is None:
                total_seg_iou = iou
            else:
                total_seg_iou += iou
            
            if total_seg_iou_wo_threshold is None:
                total_seg_iou_wo_threshold = iou_wo_threshold
            else:
                total_seg_iou_wo_threshold += iou_wo_threshold
            
            if total_seg_count is None:
                total_seg_count = count
            else:
                total_seg_count += count
            
            logging(f"{self.save_dir}/ious", count, iou, len(part_names),part_names, fname=k)
            logging(f"{self.save_dir}/ious_wo_threshold", count, iou_wo_threshold, len(part_names),part_names, fname=k)
        logging(f"{self.save_dir}/ious", total_seg_count, total_seg_iou, len(part_names),part_names, fname="miou")
        logging(f"{self.save_dir}/ious_wo_threshold", total_seg_count, total_seg_iou_wo_threshold, len(part_names),part_names, fname="miou")
    
    def log_iou(self, dict, epoch, iou_dict, intv=1):
        """
            dict : computed iou for each object in the epoch
            iou_dict : dictionary that log iou for whole epochs. Continuously updated
            intv : epoch interval. Exist for logging the test result for each N epoch. Not used for train logging.
        """

        os.makedirs(f"{self.save_dir}/ious",exist_ok=True)
        mean_miou = 0
        mean_miou_without_threshold = 0

        for k,v in dict.items():
            if k == "epoch":
                continue

            if k not in iou_dict:
                iou_dict[k] = {}
                for k_, v_ in v.items():
                    if "miou" in k_:
                        iou_dict[k][k_] = np.zeros((self.epoch//intv))
                    else:
                        iou_dict[k][k_] = np.zeros((self.epoch//intv, len(v_)))

            for k_, v_ in v.items():
                iou_dict[k][k_][epoch//intv-1] = v_
                if k_ == 'miou':
                    mean_miou += v_
                    # print(f"{k},{v_}")
                elif k_ == 'miou_without_threshold':
                    mean_miou_without_threshold += v_
        mean_miou /= len(dict.keys())
        mean_miou_without_threshold /= len(dict.keys())
        return iou_dict, mean_miou, mean_miou_without_threshold

    def plot_iou_dict(self,iou_dict, intv=1, plot_dir="plots", iou_dir="iou_txt", test=False):
        
        os.makedirs(f"{self.save_dir}/{plot_dir}",exist_ok=True)
        os.makedirs(f"{self.save_dir}/{iou_dir}", exist_ok=True)

        x  = np.arange(self.epoch//intv)
        miou_sum = np.zeros(self.epoch//intv)
        miou_sum_wo_threshold = np.zeros(self.epoch//intv)
        mriou_sum = np.zeros(self.epoch//intv)
        for k,v in iou_dict.items():
            if k == "epoch":
                continue

            category = k.split("_")[0]
            fname = k.split("_")[1]
            part_names = self.meta[category]
            
            iou = v['ious']
            iou_wo_threshold = v['ious_without_threshold']
            count = v['count']
            riou = v["rious"][:,1:]
            miou = v['miou']
            miou_wo_threshold = v['miou_without_threshold']

            miou_sum += miou
            miou_sum_wo_threshold += miou_wo_threshold
            mriou_sum += riou.mean(-1)


            fig, axs = plt.subplots(1,len(part_names)+1, squeeze=False)
            fig.set_figwidth(5*len(part_names))
            fig.set_figheight(4)


            for i in range(len(part_names)):
                axs[0,i].plot(x, iou[:,i], label="IoU")
                axs[0,i].plot(x, iou_wo_threshold[:,i], label="IoU w/o threshold")
                axs[0,i].plot(x, 1-riou[:,i], label="1-RIoU")
                axs[0,i].set_title(f"{part_names[i]}_IoU")
                axs[0,i].legend()
            axs[0, len(part_names)].plot(x, miou, label="mIoU")
            axs[0, len(part_names)].plot(x, miou_wo_threshold, label="mIoU w/o threshold")
            axs[0, len(part_names)].set_title(f"mIoU")
            axs[0, len(part_names)].legend()

            # plt.legend()
            plt.savefig(f"{self.save_dir}/{plot_dir}/{k}.png")
            plt.clf()
            plt.close()

            np.savez_compressed(f"{self.save_dir}/{iou_dir}/{k}.npz", iou=iou, iou_wo_threshold=iou_wo_threshold, count=count, riou=riou, miou=miou, miou_wo_threshold=miou_wo_threshold)

            logging(f"{self.save_dir}/{iou_dir}", count[-1], iou[-1], len(part_names),part_names, fname=k)
            logging(f"{self.save_dir}/{iou_dir}", count[-1], iou_wo_threshold[-1], len(part_names),part_names, fname=f"{k}_wo_threshold")

        num_data = len(iou_dict.keys())
        miou_mean = miou_sum/num_data
        miou_mean_wo_threshold = miou_sum_wo_threshold/num_data
        mriou_mean = mriou_sum/num_data
        plt.plot(x, miou_mean, label='mIoU mean')
        plt.plot(x, miou_mean_wo_threshold, label='mIoU mean w/o threshold')
        plt.plot(x, mriou_mean, label='mRIoU mean')
        plt.legend()
        plt.savefig(f"{self.save_dir}/{plot_dir}/miou.png")
        plt.clf()
        plt.close()

        logging(f"{self.save_dir}/{iou_dir}", [1,1,1], [miou_mean[-1], miou_mean_wo_threshold[-1], mriou_mean[-1]], 3, ['mIoU', 'mIoU_wo_threshold', 'mRIoU'], fname="miou",with_miou=False)


    def create_dir(self):
        self.save_dir=datetime.now().strftime(
                f"{self.args.save_dir}/%m-%d_%H-%M-%S"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        with open(f'{self.save_dir}/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        os.system(f"cp training_bbox_weight_whole_class.py {self.save_dir}/training_bbox_weight_whole_class.py")
        os.system(f"cp -r ./src {self.save_dir}/")
    

    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json"))

    if args.train:
        if len(args.category) > 0:
            partstad = PartSTAD(partnete_meta,args, category=args.category)
            partstad.train()
        else:
            partstad = PartSTAD(partnete_meta,args)
            partstad.train()
    elif args.test:
        partstad = PartSTAD(partnete_meta,args,eval=True)
        if len(args.category) > 0:
            partstad.eval(args.category)
        else:
            for i in range(args.start, args.end):
                partstad.eval(all_categories[i])