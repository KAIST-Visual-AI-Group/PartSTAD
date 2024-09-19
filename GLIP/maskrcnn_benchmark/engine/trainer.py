# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time
import numpy as np

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.partslip_src.sp_utils import log_iter, plot_iou_dict, miou_from_log
from .inference import inference
import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    
    print("start_iter", start_iter)
    print("len dataloader", len(data_loader))
    input("ckpt trainer line94")
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        # print("idxs : ", idxs)
        # print("images type and shape", type(images), images.tensors.shape)
        # print("targets type and len", type(targets), len(targets))
        # print("positive map",positive_map)
        # print("greenlight_map", greenlight_map)
        # print("targets", targets)
        input("ckpt1 trainer.py lin 96")
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            #PartSLIP here
            with autocast():
                if len(captions) > 0:
                    loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False
                    )
                    if is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result
                    checkpointer.save("model_best", **arguments)
                print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                    break

        if iteration % checkpoint_period == 0:
            weight = model.null_label_weight.data
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


# HJ add
def do_train_set(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    # print("part names", cfg.HJ.PART_NAMES)
    # print("checkpoint_period", checkpoint_period)
    # print("max iter :", max_iter)
    # input("ckpt trainer.py line 395")
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH

    # print("checkpoint_period", checkpoint_period)
    # input("ckpt trainer.py line 416")
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    
    # print("start_iter", start_iter)
    # print("len dataloader", len(data_loader))
    # input("ckpt trainer line426")

    iou_dict_for_all_epoch = {}
    test_iou_dict_for_all_epoch = {}
    iter_per_epoch = 1
    curr_epoch = 0
    test_epoch_intv = 5
    best_test_miou = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH #TODO
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map, extra_dict) in enumerate(data_loader, start_iter):      
        print("iteration:", iteration)
        # print("idxs : ", idxs)
        # print("images type and shape", type(images), images.tensors.shape)
        # print("targets type and len", type(targets), len(targets))
        # print("positive map",positive_map)
        # print("greenlight_map", greenlight_map)
        # print("targets", targets)
        # input("ckpt1 trainer.py line 434")
        
        # nnegative = sum(len(target) < 1 for target in targets)
        # nsample = len(targets)
        # if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
        #     logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
        #                 format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
        #     continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # print("captions", captions)
        # input("ckpt trainer.py line 461")
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            #PartSLIP here
            with autocast():
                if len(captions) > 0:
                    #PartSLIP here
                    if cfg.HJ.RETURN_CENTERNESS:
                        loss_dict, centerness, iou_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map, extra_dict=extra_dict)
                        iou_dict_for_all_epoch = log_iter(iou_dict, iou_dict_for_all_epoch, curr_epoch, max_epoch, len(cfg.HJ.PART_NAMES))
                    else:
                        loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map, extra_dict=extra_dict,only_centerness=True)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # HJ add, logging
        # if (iteration % iter_per_epoch) == 0 and (iteration != 0):
        # #     iou_dict_for_all_epoch = log_iter()


                

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        
        #HJ add
        if val_data_loader and (iteration % (test_epoch_intv*iter_per_epoch)) == 0:
            model.eval()
            if is_main_process():
                print("Evaluating HJ add")
            
            with torch.no_grad():
                for i, batch in enumerate(val_data_loader):
                    #PartSLIP here
                    test_images, test_targets, test_idxs, test_positive_map, test_positive_map_eval, test_greenlight_map, test_extra_dict = batch
                    test_images = test_images.to(device)
                    captions = None
                    try:
                        test_targets = [test_target.to(device) for test_target in test_targets]
                        captions = [t.get_field("caption") for t in test_targets if "caption" in t.fields()]
                    except:
                        pass
                    if len(captions) > 0:
                        #PartSLIP here
                        test_iou_dict = model(test_images, test_targets, captions, test_positive_map, greenlight_map = test_greenlight_map, extra_dict=test_extra_dict, return_only_iou_dict=True,vldyhead_only_return_centerness=True)
                        test_iou_dict_for_all_epoch = log_iter(test_iou_dict, test_iou_dict_for_all_epoch, curr_epoch//test_epoch_intv, max_epoch//test_epoch_intv, len(cfg.HJ.PART_NAMES))
                
                test_miou = miou_from_log(test_iou_dict_for_all_epoch)

                if test_miou > best_test_miou:
                    weight = model.null_label_weight.data
                    np.save(f"{cfg.OUTPUT_DIR}/null_weight_best.npy", np.array([weight.item()]))
                    checkpointer.save("model_best", **arguments)
                    best_test_miou = test_miou
                
                
                    
                                
            model.train()
        
        # if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
        #     if is_main_process():
        #         print("Evaluating")
        #     eval_result = 0.0
        #     model.eval()
        #     # Original function
        #     # if cfg.SOLVER.TEST_WITH_INFERENCE:
        #     #     with torch.no_grad():
        #     #         try:
        #     #             _model = model.module
        #     #         except:
        #     #             _model = model
        #     #         _result = inference(
        #     #             model = _model,
        #     #             data_loader = val_data_loader,
        #     #             dataset_name="val",
        #     #             device=device,
        #     #             expected_results=cfg.TEST.EXPECTED_RESULTS,
        #     #             expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        #     #             output_folder=None,
        #     #             cfg=cfg,
        #     #             verbose=False
        #     #         )
        #     #         if is_main_process():
        #     #             eval_result = _result[0].results['bbox']['AP']
        #     # else:
        #     #     results_dict = {}
        #     #     cpu_device = torch.device("cpu")
        #     #     for i, batch in enumerate(val_data_loader):
        #     #         images, targets, image_ids, positive_map, *_ = batch
        #     #         with torch.no_grad():
        #     #             images = images.to(device)
        #     #             if positive_map is None:
        #     #                 output = model(images)
        #     #             else:
        #     #                 captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        #     #                 output = model(images, captions, positive_map)
        #     #             output = [o.to(cpu_device) for o in output]
        #     #         results_dict.update(
        #     #             {img_id: result for img_id, result in zip(image_ids, output)}
        #     #         )
        #     #     all_predictions = all_gather(results_dict)
        #     #     if is_main_process():
        #     #         predictions = {}
        #     #         for p in all_predictions:
        #     #             predictions.update(p)
        #     #         predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
        #     #         eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
        #     #                                 box_only=cfg.DATASETS.CLASS_AGNOSTIC)
        #     #         if cfg.DATASETS.CLASS_AGNOSTIC:
        #     #             eval_result = eval_result.results['box_proposal']['AR@100']
        #     #         else:
        #     #             eval_result = eval_result.results['bbox']['AP']

        #     model.train()

        #     if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
        #         model_ema.ema.eval()
        #         results_dict = {}
        #         cpu_device = torch.device("cpu")
        #         for i, batch in enumerate(val_data_loader):
        #             images, targets, image_ids, positive_map, positive_map_eval = batch
        #             with torch.no_grad():
        #                 images = images.to(device)
        #                 if positive_map is None:
        #                     output = model_ema.ema(images)
        #                 else:
        #                     captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        #                     output = model_ema.ema(images, captions, positive_map)
        #                 output = [o.to(cpu_device) for o in output]
        #             results_dict.update(
        #                 {img_id: result for img_id, result in zip(image_ids, output)}
        #             )
        #         all_predictions = all_gather(results_dict)
        #         if is_main_process():
        #             predictions = {}
        #             for p in all_predictions:
        #                 predictions.update(p)
        #             predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
        #             eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
        #                                       box_only=cfg.DATASETS.CLASS_AGNOSTIC)
        #             if cfg.DATASETS.CLASS_AGNOSTIC:
        #                 eval_result = eval_result.results['box_proposal']['AR@100']
        #             else:
        #                 eval_result = eval_result.results['bbox']['AP']
                
        #     arguments.update(eval_result=eval_result)

        #     if cfg.SOLVER.USE_AUTOSTEP:
        #         eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
        #         # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
        #         scheduler.step(eval_result)

            
        #     if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        #         if eval_result < previous_best:
        #             patience_counter += 1
        #         else:
        #             patience_counter = 0
        #             previous_best = eval_result
        #             checkpointer.save("model_best", **arguments)
        #         print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
        #         if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
        #             if is_main_process():
        #                 print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
        #             break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            weight = model.null_label_weight.data
            np.save(f"{cfg.OUTPUT_DIR}/null_weight_{iteration:07d}.npy", np.array([weight.item()]))
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            weight = model.null_label_weight.data
            np.save(f"{cfg.OUTPUT_DIR}/null_weight_final.npy", np.array([weight.item()]))
            break

        #HJ add
        if (iteration % iter_per_epoch == 0) and (iteration != 0):
            curr_epoch += 1

        if iteration == 0 and iter_per_epoch == 1:
            curr_epoch += 1

    #HJ add
    plot_iou_dict(iou_dict_for_all_epoch, cfg.OUTPUT_DIR, cfg.HJ.PART_NAMES, max_epoch, category_iou=cfg.HJ.CATEGORY_MIOU)
    plot_iou_dict(test_iou_dict_for_all_epoch, cfg.OUTPUT_DIR, cfg.HJ.PART_NAMES, max_epoch//test_epoch_intv, plot_dir='test_plots', iou_dir='test_iou_txt')

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
