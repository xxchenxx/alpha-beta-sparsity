'''
iterative pruning for supervised task
with lottery tickets or pretrain tickets
support datasets: cifar10, Fashionmnist, cifar100, svhn
'''

import os
import pdb
from sched import scheduler
import time
import pickle
import random
import shutil
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import *
from pruning_utils import check_sparsity,extract_mask,prune_model_custom
import copy
import torch.nn.utils.prune as prune
from utils import train_psgd

def pruning_model(model, px=0.2):

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))



    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### data setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset[cifar10&100, svhn, fmnist')

##################################### model setting #################################################

##################################### basic setting #################################################
parser.add_argument('--gpu', type=int, default=None, help='gpu device id')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--resume', type=str, default=None, help='checkpoint file')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=95, type=int, help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=1, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt, rewind_lt or pt_trans)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:35506', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument("--warmup", default=0)

def main():
    best_sa = 0
    args = parser.parse_args()
    print(args)

    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('*'*50)
    print('Pruning type: {}'.format(args.prune_type))

    #torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    #args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.distributed = True
    args.multiprocessing_distributed=True

    ngpus_per_node = torch.cuda.device_count()
    if False:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model

    model = models.resnet18(pretrained=True)
    if args.checkpoint and not args.resume:
        print(f"LOAD CHECKPOINT {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        try:
            state_dict = checkpoint['state_dict']
        except KeyError:
            state_dict = checkpoint
        load_state_dict = {}
        start_state = checkpoint['state'] if 'state' in checkpoint else 0
        if start_state:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask, conv1=True)
            check_sparsity(model)
        for name in state_dict:
             if name.startswith("module."):
                   load_state_dict[name[7:]]=state_dict[name]
             else:
                   load_state_dict[name] = state_dict[name]
        model.load_state_dict(load_state_dict)
    model.fc = nn.Linear(512, 200)
    from torch.nn import init
    init.kaiming_normal_(model.fc.weight.data)

    # init pretrianed weight
    ticket_init_weight = deepcopy(model.state_dict())

    print('dataparallel mode')
    
    model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
    model = torch.nn.DataParallel(model)

    # Data loading code
    initialization = copy.deepcopy(model.module.state_dict())
    cudnn.benchmark = True
    from cub import cub200
    train_transform_list = [
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    test_transforms_list = [
            transforms.Resize(int(448/0.875)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))

    ]
    train_dataset = cub200(args.data, True, transforms.Compose(train_transform_list))
    val_dataset = cub200(args.data, False, transforms.Compose(test_transforms_list))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            start_epoch=args.start_epoch
            all_result=checkpoint['result']
            best_sa = checkpoint['best_sa']
            start_state = checkpoint['state']
            if start_state:
                current_mask = extract_mask(checkpoint['state_dict'])
                prune_model_custom(model.module, current_mask, conv1=True)
                args.epochs = 45

            model.load_state_dict(checkpoint['state_dict'])
            check_sparsity(model.module)

            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            checkpoint['optimizer']['param_groups'][0]['lr'] = args.lr
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("=> loaded checkpoint '{}' (state {}, epoch {})"
                .format(args.resume, checkpoint['state'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        start_epoch = 0
        start_state = 0
    print('######################################## Start Standard Training Iterative Pruning ########################################')
    p = torch.load('res18_CUB200_P_60.pth.tar')
    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        best_sa = 0
        check_sparsity(model, True)
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])

            acc = train_psgd(train_loader, model, criterion, optimizer, epoch, args, p)

            schedule.step()
            # evaluate on validation set
            tacc = test(val_loader, model, criterion, args)
            # evaluate on test set
            all_result['train'].append(acc)
            all_result['ta'].append(tacc)


            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'state': state,
                    'result': all_result,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_sa': best_sa,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': schedule.state_dict(),
                }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir, filename=f'epoch_{epoch}.pth.tar')

            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        #report result
        check_sparsity(model, True)
        print('* best SA={}'.format(all_result['ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0

        pruning_model(model.module, 0.2)
        check_sparsity(model.module, True)

        optimizer = torch.optim.SGD(model.parameters(), 1e-5,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        args.epochs = 45

if __name__ == '__main__':
    main()