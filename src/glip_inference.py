import os
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_model(config_file, weight_file):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    return glip_demo

def draw_rectangle(img, x0, y0, x1, y1):
    color = np.random.rand(3) * 255
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

def glip_save(pred, img_dir, save_dir, part_names, num_views=10):
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (img_dir, i))
        for j in range(len(pred)):
            if pred[j]['image_id'] == i:
                x0,y0,w,h = pred[j]['bbox']
                x1,y1 = x0+w, y0+h
                image = draw_rectangle(image, int(x0),int(y0),int(x1),int(y1))
        
        plt.imsave("%s/%d.png" % (pred_dir, i), image)

def glip_inference(glip_demo, save_dir, part_names, num_views=10,
                    save_pred_img=True, save_individual_img=False, save_pred_json=False):
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    predictions = []
    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (save_dir, i))
        result, top_predictions, _ = glip_demo.run_on_web_image(image, part_names, 0.5) 
        if save_pred_img:   
            plt.imsave("%s/%d.png" % (pred_dir, i), result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        if save_individual_img:
            save_individual_img(image, bbox, labels, len(part_names), pred_dir, i)
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            predictions.append({"image_id" : i,
                                "category_id" : labels[j].item(),
                                "bbox" : [x1,y1, x2-x1, y2-y1],
                                "score" : score[j].item()})
    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions, outfile)
    return predictions

def glip_inference_from_imgdir(glip_demo, img_dir, save_dir, part_names, num_views=10,
                    save_pred_img=True, save_individual_img=False, save_pred_json=False):
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    predictions = []
    for i in range(num_views):
        image = load_img(f"{img_dir}/{i}.png")
        result, top_predictions, _ = glip_demo.run_on_web_image(image, part_names, 0.5) 
        if save_pred_img:   
            plt.imsave("%s/%d.png" % (pred_dir, i), result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        if save_individual_img:
            save_individual_img(image, bbox, labels, len(part_names), pred_dir, i)
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            predictions.append({"image_id" : i,
                                "category_id" : labels[j].item(),
                                "bbox" : [x1,y1, x2-x1, y2-y1],
                                "score" : score[j].item()})
    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions, outfile)
    return predictions


def glip_inference_feat_extract(glip_demo, img_dir, save_dir, part_names, num_views=10,
                    save_pred_img=True, save_individual_img_=False, save_pred_json=False, save_only_centerness=False, not_post_process=False, threshold=0.5, return_centerness=False):
    """
        save_only_centerness: save only 'whole' centerness (X,X), not centerness value for each bounding box. 
    """
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    predictions = []
    features = []
    fused_features = []
    logits = []
    centernesses = []

    centerness_whole = []

    visual_features_list = []
    fused_visual_features_list = []
    for i in range(num_views):
        try:
            image = load_img(f"{img_dir}/{i}.png")
        except:
            continue
        # result, top_predictions, visual_features, centerness, fused_visual_features = glip_demo.run_on_web_image(image, part_names, thresh=threshold, return_centerness=True, vldyhead_only_return_centerness=save_only_centerness, not_post_process=not_post_process)
        if return_centerness:
            result, top_predictions, visual_features, centerness, fused_visual_features = glip_demo.run_on_web_image(image, part_names, thresh=threshold, return_centerness=return_centerness, vldyhead_only_return_centerness=save_only_centerness, not_post_process=not_post_process)
        else:
            result, top_predictions, visual_features, fused_visual_features = glip_demo.run_on_web_image(image, part_names, thresh=threshold, return_centerness=return_centerness, vldyhead_only_return_centerness=save_only_centerness, not_post_process=not_post_process)

        visual_features_list.append(visual_features[-1].cpu().numpy())
        fused_visual_features_list.append(fused_visual_features[-1].cpu().numpy())

        if save_only_centerness:
            new_centerness = []
            for j in range(len(centerness)):
                new_centerness.append(centerness[0][0,0].cpu().numpy())
            centerness_whole.append(new_centerness)
            continue
        
        #top_predictions : BoxList type
        if save_pred_img:   
            plt.imsave("%s/%d.png" % (pred_dir, i), result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        pos = top_predictions.get_field("pos").cpu().numpy().astype(np.int64)
        level = top_predictions.get_field("level").cpu().numpy().astype(np.int64)
        logit_before_sigmoid = top_predictions.get_field("logit_before_sigmoid").cpu().numpy()
        centerness_before_sigmoid = top_predictions.get_field("centerness_before_sigmoid").cpu().numpy()
        if save_individual_img_:
            save_individual_img(image, bbox, labels, len(part_names), pred_dir, i)
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            px, py = pos[j]
            curr_level = level[j]
            feature = visual_features[curr_level][:,px,py]
            fused_feature = fused_visual_features[curr_level][:,px,py]
            logit = logit_before_sigmoid[j]
            centerness = centerness_before_sigmoid[j]
            px,py,curr_level = int(px), int(py), int(curr_level)
            predictions.append({"image_id" : i,
                                "category_id" : labels[j].item(),
                                "bbox" : [x1,y1, x2-x1, y2-y1],
                                "score" : score[j].item(),
                                "pos":[px,py],
                                "level":curr_level})
            features.append(feature)
            fused_features.append(fused_feature)
            logits.append(logit)
            centernesses.append(centerness)

        


    if save_only_centerness:
        np.savez(f"{pred_dir}/glip_centerness.npz", centerness=centerness_whole)
        return None

    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions, outfile)
            features = np.stack(features,0)
            fused_features = np.stack(fused_features,0)

            fused_visual_features = [f.cpu().numpy() for f in fused_visual_features]
            visual_features = [f.cpu().numpy() for f in visual_features]
            np.savez_compressed(f"{pred_dir}/glip_feature.npz",feature=features, logit=logits, centerness=centernesses, fused_feature=fused_features)
            # for i in range(0,5):
            #     np.savez_compressed(f"{pred_dir}/image_feature.npz", visual_feature=visual_features[i], fused_visual_feature=fused_visual_features[i])
                # input("ckpt")

            visual_features = np.stack(visual_features_list,0)
            fused_visual_features = np.stack(fused_visual_features_list,0)
            np.savez_compressed(f"{pred_dir}/image_feature.npz", visual_feature=visual_features, fused_visual_feature=fused_visual_features)

    
    return predictions