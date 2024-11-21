from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, get_auc, get_f1
from util import set_optimizer
from util import *
from networks.resnet_big import SupConResNet, LinearClassifier
import torchvision.models as models
import torch.nn.functional as F
sys.path.append('/n/groups/patel/caiwei/2024_MRI/model')
from SimCLR import SimCLR  
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import wandb
from EarlyStopper import EarlyStopper


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--n_cls', type=int, default=None, help='number of classes')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()
    return opt

def process_args(opt):
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def set_model(opt):
    # model = SupConResNet(name=opt.model)
    resnet = models.resnet50(pretrained=False)
    model = SimCLR(resnet)
    # criterion = torch.nn.CrossEntropyLoss()
    class_weights = 1. / torch.tensor([1190, 1519, 5601], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    classifier = LinearClassifier(input_dim=128, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    # state_dict = ckpt['model']
    state_dict = ckpt

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    gt_labels = np.array([])
    pred_labels = np.array([])

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            # features = model.encoder(images)
            features = model(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        gt_labels = np.concatenate((gt_labels, labels.cpu().numpy()))
        pred_labels = np.concatenate((pred_labels, get_pred_label(output).cpu().numpy()))

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1_val=top1.val.item(), top1_avg=top1.avg.item()))
            sys.stdout.flush()
    f1 = f1_score(gt_labels, pred_labels, average="macro")
    # avg_auc, auc_scores = get_auc(gt_labels, pred_labels)
    return losses.avg, top1.avg.item(), f1

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    gt_labels = np.array([])
    pred_labels = np.array([])

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            # output = classifier(model.encoder(images))
            output = classifier(model(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            gt_labels = np.concatenate((gt_labels, labels.cpu().numpy()))
            pred_labels = np.concatenate((pred_labels, get_pred_label(output).cpu().numpy()))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1_val=top1.val.item(), top1_avg=top1.avg.item()))

    print(' * Acc@1 {top1_avg:.3f}'.format(top1_avg=top1.avg.item()))
    f1 = f1_score(gt_labels, pred_labels, average="macro")
    # macro_ovr_auc, weighted_ovr_auc = get_auc(gt_labels, pred_labels)
    return losses.avg, top1.avg.item(), f1

def sweep(opt):    
    opt.learning_rate = wandb.config.lr
    opt.lr_decay_epochs = wandb.config.lr_decay_epochs
    opt.lr_decay_rate = wandb.config.lr_decay_rate
    opt.weight_decay = wandb.config.weight_decay
    opt.momentum = wandb.config.momentum
    opt.epochs = wandb.config.epochs
    opt.batch_size = wandb.config.batch_size
    opt.size = wandb.config.size
    return opt

def main():
    best_f1 = 0
    opt = parse_option()
    wandb.init(project="simclr_linear_sweep_fixed", config=opt)
    opt = sweep(opt)
    opt = process_args(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    early_stopper = EarlyStopper(patience=15, min_delta=0.01)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc, f1 = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, loss: {:.2f}, accuracy:{:.2f}, f1:{:2f}'.format(
            epoch, time2 - time1, loss, acc, f1))
        wandb.log({"Epoch": epoch, "Train Loss": loss, "Train Accuracy": acc,
                   "learning_rate": optimizer.param_groups[0]['lr'],
                   "Training f1": f1})
        # eval for one epoch
        loss, val_acc, val_f1= validate(val_loader, model, classifier, criterion, opt)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'save/linear_best_sweep_model.pth')
        print('Val epoch {}, loss: {:.2f}, accuracy:{:.2f}, f1:{:2f}'.format(
            epoch, loss, val_acc, f1))

        if early_stopper.early_stop(f1):             
            break
        
        wandb.log({"Validation Loss": loss, "Validation Accuracy": val_acc,
        "Validation f1": val_f1})

    print('best f1: {:.2f}'.format(best_f1))
    torch.save(model.state_dict(), 'save/linear_last_sweep_model.pth')

if __name__ == '__main__':
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "Validation Loss"},
        "parameters": {
            "lr": {"max": 0.2, "min": 1e-5},
            "lr_decay_epochs": {"values": ['30,50,90', '10,20,50', '10,20,30,40,50,60,70,80,90']},
            # "lr": {"values": [0.1, 0.01, 0.001, 0.0001, 1e-5]},
            # "lr_decay_epochs": {"values": ["10,20,30,40,50,60,70,80,90"]},
            "lr_decay_rate": {"max": 0.5, "min": 0.1},
            "weight_decay": {"max": 0.5, "min": 0.1},
            "momentum": {"max": 0.99, "min": 0.8},
            "epochs": {"values":[500]},
            "batch_size": {"values": [64, 128, 256]},
            "size": {"values": [128, 224]}
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="simclr_linear_sweep_fixed")
    wandb.agent(sweep_id, function=main, count=20)
