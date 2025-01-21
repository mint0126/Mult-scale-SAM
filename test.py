import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from importlib import import_module
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.metric import Evaluator
# from remote_dataset.uavid_dataset import *

# from remote_dataset.vaihingen_dataset_ori import *
# from remote_dataset.potsdam_dataset_ori import *
# from remote_dataset.loveda_dataset import *

from remote_dataset.uavid_dataset import *
from CMTFNet import CMTFNet
from FTUNetFormer import ft_unetformer
from DCSwin import dcswin_base
# from R3SMamba import RS3Mamba, load_pretrained_ckpt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


'''def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb'''
    


'''def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb'''


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument("--rgb", help="whether output rgb images", action='store_true')
    parser.add_argument('--output_dir', type=str, default='/data/shanjuan/home/results/SAM-LST-main/results/vaihinwst1/test')
    parser.add_argument('--dataset', type=str,
                    default='uavid', help='dataset_name')
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default="d4", choices=[None, "d4", "lr"])
    parser.add_argument('--num_classes', type=int,
                    default=len(CLASSES), help='output channel of network')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size of the network')

    parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=4, help='total gpu')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/data/shanjuan/pretrained_models/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--my_ckpt', type=str, default='/data/shanjuan/SAM-LST-main/results/uavid_SAM_finetune8_1024_pretrain_vit_b_epo100_bs1_lr0.0003/epoch_980.740304306230569.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')

    parser.add_argument('--module', type=str, default='my_sam_LST')
    args = parser.parse_args()
    seed_everything(42)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pkg = import_module(args.module)
    model = torch.nn.DataParallel(pkg.SAM_LST()).cuda()
    
    # model = torch.nn.DataParallel(CMTFNet(num_classes=args.num_classes)).cuda()
    # model = torch.nn.DataParallel(ft_unetformer(num_classes=args.num_classes, decoder_channels=256)).cuda()
    # model = torch.nn.DataParallel(dcswin_base(num_classes=args.num_classes, pretrained=True, weight_path='/data/shanjuan/pretrained_models/stseg_base.pth')).cuda()

    # model = torch.nn.DataParallel(RS3Mamba(num_classes=args.num_classes)).cuda()
    # net = load_pretrained_ckpt(net)
    with open(args.my_ckpt, "rb") as f:
            state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    model.eval()
    evaluator = Evaluator(num_class=args.num_classes)
    evaluator.reset()
    '''if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)'''
    # test_dataset = VaihingenDataset(data_root='/data/shanjuan/remote_data/vaihingen/test', transform=val_aug)
    # test_dataset = PotsdamDataset(data_root='/data/shanjuan/remote_data/potsdam/test', transform=val_aug)
    # test_dataset = loveda_val_dataset
    test_dataset = UAVIDDataset(data_root='/data/shanjuan/remote_data/uavid/uavid_val', img_dir='images', mask_dir='masks', mode='val', mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        if args.num_classes > 1:
            multimask_output = True
        else:
            multimask_output = False

        t0 = time.time()
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions, feats = model(input['img'].cuda(), multimask_output, args.img_size)
            # raw_predictions = model(input['img'].cuda())
            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                # feat = feats[i][0:3].cpu() # .numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                
                # 假设我们有一个特征图output，其形状为(batch_size, channels, height, width)
                # output = ...
                # 将特征图转换为PIL Image对象
                # transform = transforms.ToPILImage()
                # img = transform(feat)
                # 保存图片文件
                # img.save('feat1/output_{}.png'.format(mask_name))
                # save_image(img, 'output_{}.png'.format(mask_name))
                
                results.append((mask, str(args.output_dir + mask_name), args.rgb))
    t1 = time.time()
    img_infer_time = t1 - t0
    print('images inference spends: {} FPS'.format(len(test_loader)/img_infer_time))
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(CLASSES, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    if args.dataset == "vaihingen" or args.dataset == "potsdam": 
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
    else:
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
