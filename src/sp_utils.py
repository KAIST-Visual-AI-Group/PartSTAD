import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm, trange

import imageio as io
import time
import matplotlib.pyplot as plt
import os
import cv2

cmap = plt.get_cmap('jet')

# import open3d as o3d
from PIL import Image

def plot_weight(weight,save_dir):
    num_iter, num_bbox = weight.shape

    num_bbox = min(20,num_bbox)

    weight_sigmoid = F.sigmoid(torch.from_numpy(weight)).numpy()



    fig, axs = plt.subplots(1,num_bbox, squeeze=False)
    fig.set_figwidth(4*num_bbox)
    fig.set_figheight(4)
    x = np.arange(num_iter)
    for i in range(num_bbox):
        axs[0,i].plot(x, weight[:,i])
        axs[0,i].set_title(f'bbox {i}')

    plt.savefig(f"{save_dir}/weight_plot.png")
    plt.clf()

    fig, axs = plt.subplots(1,num_bbox, squeeze=False)
    fig.set_figwidth(4*num_bbox)
    fig.set_figheight(4)
    x = np.arange(num_iter)
    for i in range(num_bbox):
        axs[0,i].plot(x, weight_sigmoid[:,i])
        axs[0,i].set_title(f'bbox {i}')

    plt.savefig(f"{save_dir}/weight_plot_sigmoid.png")
    plt.clf()
    plt.close()


def plot_ious(iou_save, iou_save_without_threshold, riou_save, riou_loss_save, save_dir, initial_ious):
    num_iter, num_label = iou_save.shape #including null label at index 0
    fig, axs = plt.subplots(1,num_label, squeeze=False)
    fig.set_figwidth(4*num_label)
    fig.set_figheight(4)
    for i in range(num_label):
        x = np.arange(num_iter)
        y1 = iou_save[:,i]
        y2 = iou_save_without_threshold[:,i]
        y3 = riou_save[:,i]
        y4 = riou_loss_save
        
        axs[0,i].set_title(f"label{i}")
        axs[0,i].plot(x,y1,label='IoU')
        axs[0,i].plot(x,y2,label='IoU without threshold')
        axs[0,i].plot(x,y3,label='RIoU')
        axs[0,i].plot(x,y4,label="loss")
        if i > 0:
            axs[0,i].hlines(initial_ious[i-1],0,num_iter, label="Initial IoU", linestyle='--', color='red')
    
    plt.legend()
    plt.savefig(f"{save_dir}/iou_plot.png")
    plt.clf()
    plt.close()


def plot_score(gt, pred, num_sp, save_dir, valid_sp=None):
    if valid_sp is not None:
        gt = gt[valid_sp]
        pred = pred[:, valid_sp]
        num_sp = num_sp[valid_sp]
    
    num_sp_ratio = num_sp/num_sp.max()
    idxs = np.argsort(num_sp_ratio)[::-1][:50]
    num_iter, num_superpoint, num_label = pred.shape
    fig, axs = plt.subplots(5,10, squeeze=False)
    fig.set_figwidth(40)
    fig.set_figheight(20)
    x = np.arange(num_iter)
    for i in range(5):
        for j in range(10):
            pos = i*10+j
            idx = idxs[pos]
            axs[i,j].set_title(f'sp{idx}:{num_sp[idx]}')
            axs[i,j].plot(x, pred[:,idx,0],label="pred score (null)",color='red')
            axs[i,j].plot(x, pred[:,idx,1],label="pred score (label 1)",color='blue')
            axs[i,j].hlines(gt[idx,0],0,num_iter, label="GT score (null)", linestyle='--', color='red')
            axs[i,j].hlines(gt[idx,1],0,num_iter, label="GT score (label 1)", linestyle='--', color='blue')

    
    plt.legend()
    plt.savefig(f"{save_dir}/score_plot.png")
    plt.clf()
    plt.close()


def draw_rectangle(img, x0, y0, x1, y1, c=None):
    if c is None:
        color = np.random.rand(3) * 255
    else:
        color = np.array(c[:3])*255
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    img[y0:y1, x0:x1, :3] /= 2
    img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img

def save_individual_img(image, bbox, labels, n_cat, pred_dir, view_id):
    n = len(labels)
    result_list = [np.copy(image) for i in range(n_cat)]
    for i in range(n):
        l = labels[i] - 1
        x0, y0, x1, y1 = bbox[i]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        result_list[l] = draw_rectangle(result_list[l], x0, y0, x1, y1)
    for i in range(n_cat):
        plt.imsave("%s/%d_%d.png" % (pred_dir, view_id, i), result_list[i][:, :, [2, 1, 0]])

def overlay_boxes(image, preds, part_names, alpha=1.0, box_pixel = 8, text_size=1.5, text_pixel=2, text_offset = 10, text_offset_original = 4, use_text=True, bbox_weight=None, weight_max=None, weight_min=None):

    cmap_box = plt.get_cmap('jet')
    new_image = image.copy()
    colors = []
    for bidx, pred in enumerate(preds):
        x0,y0,w,h = pred['bbox']
        x1,y1 = x0+w, y0+h
        top_left = (int(x0),int(y0))
        bottom_right = (int(x1),int(y1))
        if (weight_max is not None) and (weight_min is not None):
            weight = np.clip((bbox_weight[bidx] - weight_min)/(weight_max - weight_min),0.1,0.9)
            c = cmap_box(weight)
        else:
            c = (1,0,0)

        c_int = tuple([int(c[0]*255),int(c[1]*255),int(c[2]*255)])
        colors.append(c_int)
        new_image = cv2.rectangle(
            new_image, tuple(top_left), tuple(bottom_right), c_int, box_pixel)

    # Following line overlays transparent rectangle over the image
    image = cv2.addWeighted(new_image, alpha, image, 1 - alpha, 0)

    if use_text:
        template = "{}:{:.2f}"
        previous_locations = []
        for bidx,pred in enumerate(preds):
            x,y,w,h = pred['bbox']
            category_id = pred['category_id']-1
            label = part_names[category_id]
            # score = pred['weight']
            score = round(bbox_weight[bidx],2)
            s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset

            cv2.putText(
                image, s, (int(x), int(y)-text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, colors[bidx], text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y)))

    return image


def overlay_boxes2(image, preds, part_names, alpha=0.2, box_pixel = 8, text_size=4.0, text_pixel=16, text_offset = 10, text_offset_original = 4, use_text=True, bbox_weight=None, weight_max=None, weight_min=None):

    cmap_box = plt.get_cmap('turbo')
    new_image = image.copy()
    colors = []
    for bidx, pred in enumerate(preds):
        x0,y0,w,h = pred['bbox']
        x1,y1 = x0+w, y0+h
        top_left = (int(x0),int(y0))
        bottom_right = (int(x1),int(y1))
        if (weight_max is not None) and (weight_min is not None):
            weight = np.clip((bbox_weight[bidx] - weight_min)/(weight_max - weight_min),0.1,0.9)
            c = cmap_box(weight)
        else:
            c = (1,0,0)

        c_int = tuple([int(c[0]*255),int(c[1]*255),int(c[2]*255)])
        colors.append(c_int)
        new_image = cv2.rectangle(
            new_image, tuple(top_left), tuple(bottom_right), c_int, box_pixel)

        
    image = new_image

    if use_text:
        template = "{}:{:.2f}"
        template = "{:.1f}"
        previous_locations = []
        for bidx,pred in enumerate(preds):
            x,y,w,h = pred['bbox']
            category_id = pred['category_id']-1
            # label = part_names[category_id]
            # score = pred['weight']
            score = round(bbox_weight[bidx],2)
            # s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")
            s = template.format(score).replace("_", " ").replace("(", "").replace(")", "")
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset

            cv2.putText(
                image, s, (int(x), int(y)-text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, colors[bidx], text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y)))

    return image

def glip_save_with_score(preds, img_dir, save_dir, part_names, dir_name = "glip_pred", num_views=10, bbox_weights=None):
    pred_dir = os.path.join(save_dir, dir_name)
    os.makedirs(pred_dir, exist_ok = True)

    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (img_dir, i))
        pred_curr_view = []
        for pred in preds:
            if pred['image_id'] == i:
                pred_curr_view.append(pred)

        overlay_boxes(image, pred_curr_view, part_names)

        plt.imsave("%s/%d.png" % (pred_dir, i), image)

def glip_save(pred, img_dir, save_dir, part_names, dir_name = "glip_pred", num_views=10, bbox_weights=None):
    pred_dir = os.path.join(save_dir, dir_name)
    os.makedirs(pred_dir, exist_ok = True)

    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (img_dir, i))
        for j in range(len(pred)):
            if pred[j]['image_id'] == i:
                c = None
                if bbox_weights is not None:
                    c = cmap(bbox_weights[j])
                x0,y0,w,h = pred[j]['bbox']
                x1,y1 = x0+w, y0+h
                image = draw_rectangle(image, int(x0),int(y0),int(x1),int(y1), c=c)
        
        plt.imsave("%s/%d.png" % (pred_dir, i), image)


def compute_riou(gt, pred, num_sp):
    """
    input : 
        gt : list of gt score vector (or label vector) of each part | (n, k) 
        pred : list of pred score vector (or label vector) of each part | (n, k) 
        num_sp : number of points contained in each superpoint | (n,)
    output: 
        riou: list riou for each part

    n : num superpoint
    k : num label
    
    """

    num_label = gt.shape[1]
    num_sp_unsq = num_sp.unsqueeze(1)
    device = gt.get_device()

    gt_norm_p1 = torch.linalg.norm(torch.abs(gt)*num_sp_unsq, ord=1, dim=0) # (k,)
    pred_norm_p1 = torch.linalg.norm(torch.abs(pred)*num_sp_unsq, ord=1, dim=0) # (k,)

    gt_dot_pred = torch.sum(gt*pred*num_sp_unsq, dim=0) # (k,)

    riou = torch.zeros(num_label).to(device)
    count = torch.zeros(num_label).to(device)
    for i in range(num_label):
        if pred_norm_p1[i] != 0:
            count[i] = 1
            riou[i] = gt_dot_pred[i]/(gt_norm_p1[i]+pred_norm_p1[i]-gt_dot_pred[i])

    return riou, count

def compute_riou_np(gt, pred, num_sp):
    """
    input : 
        gt : list of gt score vector (or label vector) of each part | (n, k) 
        pred : list of pred score vector (or label vector) of each part | (n, k) 
        num_sp : number of points contained in each superpoint | (n,)
    output: 
        riou: list riou for each part

    n : num superpoint
    k : num label

    """

    num_label = gt.shape[1]
    num_sp_unsq = num_sp[..., np.newaxis]

    gt_norm_p1 = np.linalg.norm(np.abs(gt)*num_sp_unsq, ord=1, axis=0) # (k,)
    pred_norm_p1 = np.linalg.norm(np.abs(pred)*num_sp_unsq, ord=1, axis=0) # (k,)

    gt_dot_pred = np.sum(gt*pred*num_sp_unsq, axis=0) # (k,)

    riou = np.zeros(num_label)
    count = np.zeros(num_label)
    for i in range(num_label):
        if pred_norm_p1[i] != 0:
            count[i] = 1
            riou[i] = gt_dot_pred[i]/(gt_norm_p1[i]+pred_norm_p1[i]-gt_dot_pred[i])

    return riou, count

def compute_iou(gt_label,pred_label,num_label):
    ious = np.zeros(num_label)
    count = np.zeros(num_label)
    miou = 0

    for i in range(num_label):
        gt_mask = np.where(gt_label==i,1,0)
        pred_mask = np.where(pred_label==i,1,0)
        itsc_mask = gt_mask*pred_mask
        gt_i = np.sum(gt_mask)
        pred_i = np.sum(pred_mask)
        itsc_i = np.sum(itsc_mask)
        union_i = gt_i+pred_i-itsc_i


        if gt_i != 0:
            iou = itsc_i/union_i
            miou += iou
            count[i] = 1
            ious[i] = iou

    miou /= (np.sum(count)+1e-12)

    return ious, miou, count

def compute_iou_null_label(gt_label,pred_label):
    ious = 0
    count = 0
    miou = 0

    gt_mask = np.where(gt_label==-1,1,0)
    pred_mask = np.where(pred_label==-1,1,0)
    itsc_mask = gt_mask*pred_mask
    gt_i = np.sum(gt_mask)
    pred_i = np.sum(pred_mask)
    itsc_i = np.sum(itsc_mask)
    union_i = gt_i+pred_i-itsc_i

    if gt_i != 0:
        iou = itsc_i/union_i
        miou += iou
        count = 1
        ious = iou

    return ious, count

def compute_score(superpoint, 
                  num_label, 
                  num_views,
                  device, 
                  sp_visible_cnt,
                  sp_map_list, 
                  bbox_weight, 
                  view_bbox_mask_list,
                  valid_sp,
                  update_null_weight = False,
                  use_relu_weight = True
                  ):
    sem_score = torch.zeros((len(superpoint),num_label+1)).to(device)
    sem_score[:,0] = (torch.clamp(bbox_weight[0],5e-4) if use_relu_weight else bbox_weight[0]) if update_null_weight else 0.5

    bbox_weight = bbox_weight[1:]

    for k, sp_map in enumerate(sp_map_list):
        if valid_sp[k] == 0:
            continue

        weighted_sp_map = sp_map * bbox_weight.unsqueeze(0).unsqueeze(0) # n_sp, 
        for l in range(num_views):
            view_bbox_mask = view_bbox_mask_list[k][l] # n_pred,

            apply_view_bbox_mask = weighted_sp_map * view_bbox_mask.unsqueeze(0).unsqueeze(0) # n_sp, n_cat, n_bbox

            sp_score_wrt_view = torch.sum(torch.max(apply_view_bbox_mask, dim=-1).values,dim=0) # n_sp, n_cat, n_bbox -> n_sp, n_cat -> n_cat

            sem_score[k,1:] += sp_score_wrt_view/(sp_visible_cnt[k]+1e-6)

    return sem_score


def compute_valid_label(num_label, 
                  num_views,
                  device,
                  sp_map_list,
                  view_bbox_mask_list,
                  valid_sp
                  ):
    """
        updated in 0925 for applying minus weight value
        k : sp idx
        l : view idx
        valid_label_mask[k][l][i,j] = 1 if p_i is contained at least one bbox with label j
        valid_label[k,j] = 1 if at least one point of sp_k is contained in at least one bbox with label j, otherwise 0
    """
    valid_label_mask_list = []
    valid_label = torch.zeros((len(sp_map_list),num_label)).to(device)
    for k, sp_map in enumerate(sp_map_list):
        n_sp, n_cat, _ = sp_map.shape
        if valid_sp[k] == 0:
            valid_label_mask_list.append(None)
            continue

        valid_label_mask_per_view = torch.zeros((num_views,n_sp,n_cat)).to(device)
        valid_label_sum = torch.zeros(n_cat).to(device)
        for l in range(num_views):
            view_bbox_mask = view_bbox_mask_list[k][l] # n_pred,
            view_bbox_mask = view_bbox_mask.unsqueeze(0).unsqueeze(0)
            if view_bbox_mask.sum() == 0:
                continue
            valid_label_view_l = torch.max(sp_map * view_bbox_mask, dim=-1).values # n_sp, n_cat
            valid_label_sum += torch.sum(valid_label_view_l, dim=0) # n_cat
            valid_label_mask_per_view[l] = valid_label_view_l

        valid_label_mask_list.append(valid_label_mask_per_view)
        valid_label[k,valid_label_sum>0] = 1

    return valid_label_mask_list, valid_label


def compute_valid_label_np(num_label, 
                  num_views,
                  sp_map_list,
                  view_bbox_mask_list,
                  valid_sp
                  ):
    """
        updated in 0925 for applying minus weight value
    """
    valid_label_mask_list = []
    for k, sp_map in enumerate(sp_map_list):
        n_sp, n_cat, _ = sp_map.shape
        valid_label_mask_per_view = np.zeros((num_views,n_sp,n_cat))
        if valid_sp[k] == 0:
            continue
        
        for l in range(num_views):
            view_bbox_mask = view_bbox_mask_list[k][l] # n_pred,
            view_bbox_mask = view_bbox_mask[np.newaxis,np.newaxis,...]
            

            valid_label_view_l = np.max(sp_map * view_bbox_mask, axis=-1) # n_sp, n_cat
            valid_label_mask_per_view[l] = valid_label_view_l

        valid_label_mask_list.append(valid_label_mask_per_view)
    return valid_label_mask_list


def get_sp_label(sp, label):
    """
        Assign the label which has the largest portion to each superpoint
        null label : -1
        cateogry label : 0, 1, 2, ..., n-1
    """
    sp_labels = []
    for i in range(len(sp)):
        selected_label = label[sp[i]]
        val, cnt = np.unique(selected_label, return_counts=True)
        ind = np.argmax(cnt)
        sp_labels.append(val[ind])
    
    sp_labels = np.array(sp_labels)
    return sp_labels

def get_sp_score(sp, label, num_label):
    """
        Assign the label score to each superpoint. 
        Score is computed as the portion of points which are both included in superpoint and also have the same label.
    """
    sp_score = np.zeros((len(sp),num_label+1))
    num_sp = np.zeros(len(sp))
    for i in range(len(sp)):
        selected_label = label[sp[i]]
        num_sp_i = len(selected_label)
        val, cnt = np.unique(selected_label, return_counts=True)
        for idx, j in enumerate(val): # first idx is always -1
            sp_score[i,j+1] = cnt[idx]/num_sp_i

        num_sp[i] = num_sp_i
    
    return sp_score, num_sp

def get_mask(arr):
    vals = np.unique(arr)

    if len(vals) == 1:
        return None, None

    vals = vals[1:]

    masks = []
    for i in range(len(vals)):
        masks.append(np.where(arr==vals[i], 1, 0))
    
    masks = np.stack(masks)

    return masks, vals

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image



def logging(save_dir,seg_count,seg_ious,num_label,part_names,fname="miou", with_miou=True):
    with open(f"{save_dir}/{fname}.txt","w") as f:
        # f.write(f"seg_mIoU:{seg_miou}\nins_mIoU:{ins_miou}")
        w = ""
        ious_sum = 0
        count = 0
        for j in range(num_label):
            if seg_count[j] == 0:
                w = w + f"{part_names[j]}_IoU:NaN\n"
            else:
                IoU_val = seg_ious[j]/seg_count[j]
                w = w + f"{part_names[j]}_IoU:{IoU_val}\n"
                ious_sum += IoU_val
                count += 1

        if with_miou:
            seg_miou = ious_sum/(count + 1e-12)
            w = f"seg_mIoU:{seg_miou}\n" + w

        f.write(w)


def get_miou(total_iou, total_count):
    
    num_label = total_iou.shape[1]
    iou_sum = np.zeros(total_iou.shape[0]) # N,
    count = 0
    for j in range(num_label):
        if total_count[0,j] != 0:
            count += 1
            iou_sum += total_iou[:,j]
    
    if count == 0:
        miou = 0
    else:
        miou = iou_sum / count

    return miou


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