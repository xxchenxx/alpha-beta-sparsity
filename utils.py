import os
import time
import copy
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import *
from models.resnet import resnet18, resnet50, resnet152
from pruning_utils import *
from models.resnet import MaskedConv2d

__all__ = ['setup_model_dataset', 'setup_seed',
            'train', 'test',
            'save_checkpoint', 'load_weight_pt_trans', 'load_ticket']

def setup_model_dataset(args):

    #prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar10c':
        classes = 10
        train_loader, val_loader, test_loader = cifar10c_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'cub':
        classes = 200
        train_loader, val_loader, test_loader = cub_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'imagenet':
        classes = 1000
        train_loader, val_loader, test_loader = imagenet_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'imagenetc':
        classes = 1000
        train_loader, val_loader, test_loader = imagenetc_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar100':
        classes = 100
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'svhn':
        classes = 10
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'fmnist':
        classes = 10
        train_loader, val_loader, test_loader = fashionmnist_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    else:
        raise ValueError("Unknown Dataset")

    #prepare model
    if args.arch == 'resnet18':
        model = resnet18(num_classes = classes)
    elif args.arch == 'resnet50':
        model = resnet50(num_classes = classes)
    elif args.arch == 'resnet152':
        model = resnet152(num_classes = classes)
    else:
        raise ValueError("Unknown Model")

    if args.dataset == 'fmnist':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return model, train_loader, val_loader, test_loader

def train(train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_with_imagenet(train_loader, imagenet_train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    imagenet_train_loader_iter = iter(imagenet_train_loader)
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()
        if False:
            try:
                imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
            except:
                imagenet_train_loader_iter = iter(imagenet_train_loader)
                imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
            # compute output
            imagenet_image = imagenet_image.cuda()
            imagenet_target = imagenet_target.cuda()

            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_lower()

            output_clean = model(imagenet_image)
            loss = criterion(output_clean, imagenet_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()

        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.set_upper()

        output_old, output_new = model(image)
        loss = criterion(output_new, target) + 0 * output_old.sum()

        for name, m in model.named_modules():
           if isinstance(m, MaskedConv2d):
                loss = loss + args.l1_reg_beta * torch.sum(torch.abs(m.mask_beta))
        optimizer.zero_grad()
        loss.backward()
        # remove weights grad
        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.weight.grad = None
                m.mask_alpha.grad = None
        optimizer.step()
        # calculate (a + b)
        model.zero_grad()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_new.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    if args.rank == 0:
        for name, p in model.named_parameters():
            if 'mask_alpha' in name or 'mask_beta' in name:
                print(name, (p.data.abs() ** 5).mean())
                

    return top1.avg


def train_with_imagenet_gdp(train_loader, imagenet_train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    # imagenet_train_loader_iter = iter(imagenet_train_loader)
    for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.epsilon *= 0.9
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()
        if False:
            try:
                imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
            except:
                imagenet_train_loader_iter = iter(imagenet_train_loader)
                imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
            # compute output
            imagenet_image = imagenet_image.cuda()
            imagenet_target = imagenet_target.cuda()

            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_lower()

            output_clean = model(imagenet_image)
            loss = criterion(output_clean, imagenet_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()

        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.set_upper()

        output_old, output_new = model(image)
        loss = criterion(output_new, target) + 0 * output_old.sum()
        optimizer.zero_grad()
        loss.backward()
        # remove weights grad
        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.weight.grad = None
                m.mask_alpha.grad = None
                # print(name, m.mask_beta.grad.abs().mean())
        optimizer.step()
        # calculate (a + b)
        model.zero_grad()
        loss = loss.float()
        for name, m in model.named_modules():
           if isinstance(m, MaskedConv2d):
                beta = m.mask_beta.data.detach().clone()
                lr = optimizer.param_groups[1]['lr']
                #print(beta.data.abs().mean())
                m1 = beta >= lr * args.lamb
                m2 = beta <= -lr * args.lamb
                m3 = (beta.abs() < lr * args.lamb)
                m.mask_beta.data[m1] = m.mask_beta.data[m1] - lr * args.lamb
                m.mask_beta.data[m2] = m.mask_beta.data[m2] + lr * args.lamb
                m.mask_beta.data[m3] = 0
        # measure accuracy and record loss
        prec1 = accuracy(output_new.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    if args.rank == 0:
        for name, p in model.named_parameters():
            if 'mask_alpha' in name or 'mask_beta' in name:
                print(name, (p.data.abs()).mean())
                

    return top1.avg


def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg


def test_with_imagenet(val_loader, model, criterion, args, alpha_params, beta_params):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for name, m in model.named_modules():
        if isinstance(m, MaskedConv2d):
            m.set_upper()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            _, output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

def load_weight_pt_trans(model, initalization, args):
    print('loading pretrained weight')
    loading_weight = extract_main_weight(initalization, fc=args.fc, conv1=args.conv1)

    for key in loading_weight.keys():
        if not (key in model.state_dict().keys()):
            print(key)
            assert False

    print('*number of loading weight={}'.format(len(loading_weight.keys())))
    print('*number of model weight={}'.format(len(model.state_dict().keys())))
    model.load_state_dict(loading_weight, strict=False)

def load_ticket(model, args):

    # mask
    if args.mask_dir:

        current_mask_weight = torch.load(args.mask_dir, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']
        current_mask = extract_mask(current_mask_weight)

        if args.reverse_mask:
            current_mask = reverse_mask(current_mask)
        prune_model_custom(model, current_mask, conv1=args.conv1)
        check_sparsity(model, conv1=args.conv1)
    # weight
    if args.pretrained:

        initalization = torch.load(args.pretrained, map_location=torch.device('cuda:' + str(args.gpu)))
        if args.dict_key:
            print('loading from {}'.format(args.dict_key))
            initalization = initalization[args.dict_key]

        if args.load_all:
            loading_weight = copy.deepcopy(initalization)
        else:
            loading_weight = extract_main_weight(initalization, fc=False, conv1=False)

        for key in loading_weight.keys():
            assert key in model.state_dict().keys()

        print('*number of loading weight={}'.format(len(loading_weight.keys())))
        print('*number of model weight={}'.format(len(model.state_dict().keys())))
        model.load_state_dict(loading_weight, strict=False)

def warmup_lr(epoch, step, optimizer, one_epoch_step, args):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False