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

from models.resnet import MaskedConv2d

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
parser.add_argument('--pruning_times', default=19, type=int, help='overall times of pruning')
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
parser.add_argument("--output_name", default=0)
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
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    from models.resnet import resnet18
    model = resnet18(pretrained=False, num_classes=1000)
    model.new_fc = nn.Linear(512, 200)
    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights()
    if args.checkpoint and not args.resume:
        print(f"LOAD CHECKPOINT {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint,map_location="cpu")
        state_dict = checkpoint['state_dict']
        load_state_dict = {}
        start_state = checkpoint['state']
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
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        for m in model.modules():
            if isinstance(m, MaskedConv2d):
                m.epsilon = 0.1 * (0.9) ** epoch
    from torch.nn import init
    init.kaiming_normal_(model.fc.weight.data)
    
    print('dataparallel mode')

    cudnn.benchmark = True
    model.cuda()
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    
    all_result = {}
    all_result['train'] = []
    all_result['test_ta'] = []
    all_result['ta'] = []

    start_epoch = 0
    start_state = 0
    print('######################################## Start Standard Training Iterative Pruning ########################################')

    model.eval()

    start = time.time()
    with torch.no_grad():
        representations = {i: [] for i in range(0, 200, 20)}
        for i, (image, target) in enumerate(train_loader):

            image = image.cuda()
            target = target.cuda()

            # compute output
            _, rep = model(image, True)
            for j in range(target.shape[0]):
                if target[j] % 20 == 0:
                    representations[int(target[j])].append(rep[j])
        
        labels = []
        for key in representations:
            representations[key] = torch.stack(representations[key])
            labels.extend([key] * representations[key].shape[0])
        from sklearn.manifold import TSNE

        import matplotlib.pyplot as plt
        import seaborn as sns
        s = []
        for key in representations:
            s.append(representations[key])
        s = torch.cat(s, 0).cpu().numpy()
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(s)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset = {}
    df_subset['label'] = labels
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    import pickle 
    pickle.dump(df_subset, open("result.pkl", "wb"))
    plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=1
    )
    fig = plot.get_figure()
    fig.savefig(args.output_name) 
if __name__ == '__main__':
    main()