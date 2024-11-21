from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets


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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--n_cls', type=int, default=None, help='number of classes')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--ckpt_cls', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'path':
        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return val_loader

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

    ckpt_cls = torch.load(opt.ckpt_cls, map_location='cpu')
    classifier.load_state_dict(ckpt_cls)
    
    # breakpoint()    
    return model, classifier, criterion

def get_confusion_matrix(labels, pred):
    cm = confusion_matrix(labels, pred)
    return cm

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    f1 = AverageMeter()

    pred_labels = np.array([])
    gt_labels = np.array([])

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model(images))
            loss = criterion(output, labels)
            pred = get_pred_label(output)

            cm = get_confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())
            print(cm)
            # breakpoint()
            pred_labels = np.concatenate((pred_labels, pred.cpu().numpy()))
            gt_labels = np.concatenate((gt_labels, labels.cpu().numpy()))

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            f1_score = get_f1(labels, output)
            f1.update(f1_score, bsz)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            

            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1_val:.3f} ({top1_avg:.3f})\t'
                    'F1 {f1_val:.3f} ({f1_avg:.3f}'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1_val=top1.val.item(), top1_avg=top1.avg.item(),
                    f1_val=f1.val, f1_avg=f1.avg))
        

    print(' * Acc@1 {top1_avg:.3f}'.format(top1_avg=top1.avg.item()))
    return losses.avg, top1.avg.item(), f1.avg, pred_labels, gt_labels


def main():
    opt = parse_option()

    # build data loader
    val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    loss, val_acc, f1, pred_labels, gt_labels = validate(val_loader, model, classifier, criterion, opt)

    print('Val loss: {:.2f}, accuracy:{:.2f}, f1:{:.2f}'.format(
           loss, val_acc, f1))
    
    cm = get_confusion_matrix(gt_labels, pred_labels)
    print(cm)

    f1_per_class = f1_score(gt_labels, pred_labels, average=None)

    print(f"F1 score for each class: {f1_per_class}")

    f1 = f1_score(gt_labels, pred_labels, average="macro")

    print(f"Averaged F1 score: {f1}")

if __name__ == '__main__':
    main()
