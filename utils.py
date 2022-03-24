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

def train_with_imagenet(train_loader, imagenet_train_loader, model, criterion, optimizer, epoch, args, alpha_params, beta_params):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    imagenet_train_loader_iter = iter(imagenet_train_loader)
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        backup_params = {}

        image = image.cuda()
        target = target.cuda()
        try:
            imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
        except:
            imagenet_train_loader_iter = iter(imagenet_train_loader)
            imagenet_image, imagenet_target = next(imagenet_train_loader_iter)
        # compute output

        for name, p in model.named_parameters():
            backup_params[name] = p.detach().data
            p = p * alpha_params[name]

        output_clean = model(imagenet_image)
        loss = criterion(output_clean, imagenet_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate (a + b)
        model.zero_grad()
        l1_loss = 0
        for name, p in model.named_parameters():
            alpha_params[name].grad.zero_()
            beta_params[name].grad.zero_()
            p.copy_(backup_params[name]).mul_(alpha_params[name]).mul_(beta_params[name])
            

        output_clean = model(image)
        loss = criterion(output_clean, target)
        for name, p in model.named_parameters():
            loss = loss + 0.001 * torch.sum(torch.abs(alpha_params[name] * beta_params[name]))
        output = output_clean.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            alpha_params[name].data.sub_(alpha_params[name].grad * 0.01)
            beta_params[name].data.sub_(beta_params[name].grad * 0.01)
            print(name, alpha_params[name].grad)
            print(name, beta_params[name].grad)
            alpha_params[name].grad.zero_()
            beta_params[name].grad.zero_()
            


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


def concrete_stretched(alpha, l=0., r = 1.):
    u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
    s = (torch.sigmoid(u.log() - (1-u).log() + alpha)).detach()
    u = s*(r-l) + l
    t = u.clamp(0, 1000)
    z = t.clamp(-1000, 1)
    dz_dt = (t < 1).float().to(alpha.device).detach()
    dt_du = (u > 0).float().to(alpha.device).detach()
    du_ds = r - l
    ds_dalpha = (s*(1-s)).detach()
    dz_dalpha = dz_dt*dt_du*du_ds*ds_dalpha
    return z.detach(), dz_dalpha.detach()

def train_co(train_loader, imagenet_train_loader, model, criterion, optimizer, epoch, args,  model_params):

    losses = AverageMeter()
    top1 = AverageMeter()
    per_params_z = {}
    per_params_z_grad = {}
    # switch to train mode
    
    log_ratio = np.log(-args.concrete_lower / args.concrete_upper)
    start = time.time()
    for n, p in model.named_parameters():
        if n not in model_params:
            print(" n not in model_params")
            embed()
        assert(n in model_params)
        if "fc" in n:
            nonzero_params += p.numel()
            p.data.copy_(model_params[n][0].data + model_params[n][1].data)
        else:
            if args.per_params_alpha == 1:
                params_z, params_z_grad = concrete_stretched(per_params_alpha[n], args.concrete_lower,
                        args.concrete_upper)
                per_params_z[n] = params_z
                per_params_z_grad[n] = params_z_grad

            z, z_grad = concrete_stretched(model_params[n][2], args.concrete_lower, args.concrete_upper)
            
            ind = 0
            l0_pen[ind] += torch.sigmoid(model_params[n][2] - log_ratio).sum()
            l0_pen_sum += torch.sigmoid(model_params[n][2] - log_ratio).sum()
            
            z2 =  per_params_z[n]

            grad_params[n] = [model_params[n][1] * z2, z * z2, z_grad, model_params[n][1] * z]

            if args.per_params_alpha == 1:
                l0_pen[ind] += torch.sigmoid(per_params_alpha[n] - log_ratio).sum()
        
            p.data.copy_(model_params[n][0].data + (z2*z).data*model_params[n][1].data)
            nonzero_params += ((z2*z)>0).float().detach().sum().item()

    model.train()

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

        for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                if "classifier" in n:
                    bert_params[n][1].grad.copy_(p.grad.data)
                else:
                    try:
                        bert_params[n][1].grad.copy_(p.grad.data * grad_params[n][1].data)
                    except:
                        embed()
                    bert_params[n][2].grad.copy_(p.grad.data * grad_params[n][0].data *
                                                 grad_params[n][2].data)
                
                    if args.per_params_alpha == 1:
                        per_params_alpha[n].grad.copy_(torch.sum(p.grad.data * grad_params[n][3].data * 
                                per_params_z_grad[n].data))
                    if args.per_layer_alpha == 1:
                        per_layer_alpha.grad[get_layer_ind(n)] += torch.sum(p.grad.data * grad_params[n][3].data *
                                per_layer_z_grad[ind].data)

                sum_l0_pen = 0
                for i in range(total_layers):
                    if l0_pen[i] != 0:
                        sum_l0_pen += (sparsity_pen[i] * l0_pen[i]).sum()
                sum_l0_pen.sum().backward()

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(finetune_params, args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(alpha_params, args.max_grad_norm)
                optimizer.step()
                alpha_optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                params_norm = [0, 0, 0, 0, 0, 0]
                exp_z = 0
                for n, p in bert_params.items():
                    params_norm[0] += p[2].sum().item()
                    params_norm[1] += p[2].norm().item()**2
                    params_norm[2] += p[2].grad.norm().item()**2
                    params_norm[3] += torch.sigmoid(p[2]).sum().item()
                    params_norm[4] += p[2].numel()
                    # params_norm[5] += (grad_params[n][1] > 0).float().sum().item()
                    if args.per_params_alpha == 1:
                        exp_z += (torch.sigmoid(p[2]).sum() * torch.sigmoid(per_params_alpha[n])).item()
                    else:
                        exp_z += torch.sigmoid(p[2]).sum().item()

                    p[1].grad.zero_()
                    p[2].grad.zero_()

                mean_exp_z = exp_z / params_norm[4]

                    
                if args.per_layer_alpha == 1:
                    per_layer_alpha.grad.zero_()
                if args.per_params_alpha == 1:
                    for n,p in per_params_alpha.items():
                        p.grad.zero_()


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