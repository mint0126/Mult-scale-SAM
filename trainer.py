import argparse
import logging
import os
from tools.metric import Evaluator
os.environ["CUDA_VISIBLE_DEVICES"] =  '0,1,2,3'
# from catalyst.contrib.nn import Lookahead
# from catalyst import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from loss import UnetFormerLoss
from utils import display_images, to_rgb, denormalize
import torchvision
from icecream import ic
import imageio
from losses import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils


CLASSES = ('Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car',  'Static_Car', 'Human', 'Clutter')
# CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
# CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')


def val(args, model, dataloader, metrics_val):
    model.eval()
    for _, sampled_batch in enumerate(dataloader):
        image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']  # [b, c, h, w], [b, h, w]

        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        with torch.no_grad():
            outputs = model(image_batch)

        pre_mask = nn.Softmax(dim=1)(outputs)
        pre_mask = pre_mask.argmax(dim=1)

        for i in range(label_batch.shape[0]):
            metrics_val.add_batch(label_batch[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        if args.dataset == "vaihingen":
            val_mIoU = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
            val_F1 = np.nanmean(metrics_val.F1()[:-1])
            val_OA = np.nanmean(metrics_val.OA())
        elif args.dataset == 'potsdam':
            val_mIoU = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
            val_F1 = np.nanmean(metrics_val.F1()[:-1])
            val_OA = np.nanmean(metrics_val.OA())
        else:
            val_mIoU = np.nanmean(metrics_val.Intersection_over_Union())
            val_F1 = np.nanmean(metrics_val.F1())
            val_OA = np.nanmean(metrics_val.OA())
    val_iou_per_class = metrics_val.Intersection_over_Union()
    eval_value = {'mIoU': val_mIoU,
                    'F1': val_F1,
                    'OA': val_OA}
    print('val:', eval_value)
    iou_value = {}
    for class_name, iou in zip(CLASSES, val_iou_per_class):
        iou_value[class_name] = iou
    print(iou_value)
    metrics_val.reset()

    return val_mIoU, val_F1, val_OA


def trainer_synapse(args, model, snapshot_path, trainloader, valloader):

    model.train()

    if args.AdamW:
        layerwise_params = {"backbone.*": dict(lr=args.backbone_lr, weight_decay=args.backbone_weight_decay)}
        net_params = utils.process_model_params(model, layerwise_params=layerwise_params)
        base_optimizer = torch.optim.AdamW(net_params, lr=args.base_lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(base_optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, weight_decay=args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), lr=b_lr, weight_decay=args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    else:
        # RS3Mamba
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_miou = 0.0
    calc_loss = UnetFormerLoss(ignore_index=255)
    metrics_train = Evaluator(num_class=args.num_classes)
    metrics_val = Evaluator(num_class=args.num_classes)
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:

        model.train()
        epoch_samples = 0
        for _, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']  # [b, c, h, w], [b, h, w]
            if isinstance(image_batch, list): 
                main_image_batch, _, mian_label_batch, scale_label_batch = image_batch[0].cuda(), image_batch[1].cuda(), label_batch[0].cuda(), label_batch[1].cuda()
            else:
                main_image_batch, _, mian_label_batch, scale_label_batch = image_batch.cuda(), None, label_batch.cuda(), None

            outputs = model(main_image_batch)
            if len(outputs) == 2:
                if isinstance(outputs[0], list):
                    scale_label_batch = scale_label_batch.permute(1, 0, 2, 3)
                    for i in range(len(outputs[0])):
                        pre_mask = nn.Softmax(dim=1)(outputs[0][i])
                        pre_mask = pre_mask.argmax(dim=1)
                    
                        for j in range(scale_label_batch[i].shape[0]):
                            metrics_train.add_batch(scale_label_batch[i][j].cpu().numpy(), pre_mask[j].cpu().numpy())
                    loss = calc_loss(outputs, scale_label_batch)
                else:
                    pre_mask = nn.Softmax(dim=1)(outputs[0])
                    pre_mask = pre_mask.argmax(dim=1)
                
                    for i in range(mian_label_batch.shape[0]):
                        metrics_train.add_batch(mian_label_batch[i].cpu().numpy(), pre_mask[i].cpu().numpy())
            else:
                pre_mask = nn.Softmax(dim=1)(outputs)
                pre_mask = pre_mask.argmax(dim=1)

                for i in range(mian_label_batch.shape[0]):
                    metrics_train.add_batch(mian_label_batch[i].cpu().numpy(), pre_mask[i].cpu().numpy())

            loss = calc_loss(outputs, mian_label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_samples += sampled_batch['img'][0].size(0)
            
            iter_num = iter_num + 1
          
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)


            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = main_image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                if len(outputs) == 2:
                    output_masks = outputs[0]
                else:
                    output_masks = outputs
                if isinstance(output_masks, list):
                    output_masks = output_masks[0]
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)

        if args.dataset == "vaihingen":
            train_mIoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            train_F1 = np.nanmean(metrics_train.F1()[:-1])
            train_OA = np.nanmean(metrics_train.OA())
        elif args.dataset == 'potsdam':
            train_mIoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            train_F1 = np.nanmean(metrics_train.F1()[:-1])
            train_OA = np.nanmean(metrics_train.OA())
        else:
            train_mIoU = np.nanmean(metrics_train.Intersection_over_Union())
            train_F1 = np.nanmean(metrics_train.F1())
            train_OA = np.nanmean(metrics_train.OA())
        train_iou_per_class = metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': train_mIoU,
                      'F1': train_F1,
                      'OA': train_OA}
        print('train:', eval_value)
        iou_value = {}
        for class_name, iou in zip(CLASSES, train_iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        metrics_train.reset()
    
        val_mIoU, val_F1, val_OA = val(args, model, valloader, metrics_val)

        metrics_val.reset()
        logging.info('epoch %d : train_miou: %f : train_F1 %f : train_OA : %f ' % (epoch_num, train_mIoU,  train_F1, train_OA))
        logging.info('epoch %d : val_miou %f : val_F1 %f : val_OA : %f ' % (epoch_num, val_mIoU,  val_F1, val_OA))
       
        writer.add_scalar('info/train_miou', train_mIoU, epoch_num)
        writer.add_scalar('info/val_miou', val_mIoU, epoch_num)
           
        # val
        if val_mIoU > best_miou:
            best_miou = val_mIoU
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + str(val_mIoU) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
