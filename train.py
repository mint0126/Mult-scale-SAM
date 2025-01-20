import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import random
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from remote_dataset.build_dataset import prepare_dataset
from importlib import import_module
from torchvision import transforms
import random
import sys
from segment_anything import sam_model_registry
from trainer_LST import trainer_synapse
# from trainer import trainer_synapse # 当使用UNetFormer，ft_unetformer，dcswin_base，CMTFNet时使用trainer
from icecream import ic
from remote_dataset.vaihingen_dataset_ori import * 
# from remote_dataset.potsdam_dataset_ori import * 
# from remote_dataset.loveda_dataset import *
# from remote_dataset.uavid_dataset import *
from torch.utils.data import DataLoader
from FTUNetFormer import ft_unetformer
from UNetFormer import UNetFormer
from DCSwin import dcswin_base
from CMTFNet import CMTFNet
from R3SMamba import RS3Mamba, load_pretrained_ckpt


parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str, default='/data/shanjuan/home/results/SAM-LST-main/results')
parser.add_argument('--dataset', type=str,
                    default='vaihingen', help='dataset_name')
parser.add_argument('--experiment', type=str,
        default='SAM_LST', help='experiment_name')

parser.add_argument('--num_classes', type=int,
                    default=len(CLASSES), help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=4, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0006,
                    help='segmentation network learning rate')
parser.add_argument('--backbone_lr', type=float, default=6e-5,
                    help='segmentation backbone network learning rate')
parser.add_argument('--backbone_weight_decay', type=float, default=2.5e-4,
                    help='backbone weight decay')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='weight decay')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--input_size', type=int, default=1024, help='The input size for training SAM model')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=100,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='my_sam_LST')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    dataset_name = args.dataset
    
    args.is_pretrain = True
    args.exp = dataset_name+ '_'+ args.experiment + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    pkg = import_module(args.module)
    net = pkg.SAM_LST().cuda()
  
    # net = UNetFormer(num_classes=args.num_classes).cuda()
  
    # net = ft_unetformer(num_classes=args.num_classes, decoder_channels=256).cuda()
    
    # net = dcswin_base(num_classes=args.num_classes).cuda()
  
    # net = CMTFNet(num_classes=args.num_classes).cuda()
  
    # net = RS3Mamba(num_classes=args.num_classes).cuda()
    # net = load_pretrained_ckpt(net) # RS3Mamba

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))
    batch_size = args.batch_size * args.n_gpu
    
    train_dataset = VaihingenDataset(data_root='/data/shanjuan/home/remote_data/remote_data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

    val_dataset = VaihingenDataset(data_root='/data/shanjuan/home/remote_data/remote_data/vaihingen/test', img_dir='images_1024', mask_dir='masks_1024', transform=val_aug)
    '''
    train_dataset = PotsdamDataset(data_root='/data/shanjuan/home/remote_data/remote_data/potsdam/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

    val_dataset = PotsdamDataset(data_root='/data/shanjuan/home/remote_data/remote_data/potsdam/test', transform=val_aug)

    train_dataset = LoveDATrainDataset()

    val_dataset = loveda_val_dataset

    train_dataset = UAVIDDataset(data_root='/data/shanjuan/home/remote_data/remote_data/uavid/uavid_train', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

    val_dataset = UAVIDDataset(data_root='/data/shanjuan/home/remote_data/remote_data/uavid/uavid_val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))
    '''
    trainloader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

    valloader = DataLoader(dataset=val_dataset,
                        batch_size=1,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


    if args.n_gpu > 1:
        net = torch.nn.DataParallel(net)

    trainer = {args.dataset: trainer_synapse}
    # trainer[dataset_name](args, net, snapshot_path, trainloader, valloader) # UNetFormer, CMTFNet, FT-UNetFormer
    trainer[dataset_name](args, net, snapshot_path, trainloader, valloader, multimask_output)

# python train.py --warmup --AdamW
# python train.py --AdamW
