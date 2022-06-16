# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47,

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from utils.utils import progress_bar
from utils.data_utils import get_loader

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--amp', action='store_false', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--quant', action='store_true', help='quantize model')
parser.add_argument('--K', default=4, type=int)
parser.add_argument('--net', default='resnet20')
parser.add_argument('--bs', default='512')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
args = parser.parse_args()

if args.quant:
    from network.resnet_orig import *
else:
    from network.resnet_noquant import *

# take in args
import wandb
watermark = "{}_lr{}".format(args.net, args.lr)
if args.amp:
    watermark += "_useamp"
if args.quant:
    watermark += "_quant{}".format(args.K)

wandb.init(project="pact_cim",
           name=watermark)
wandb.config.update(args)

if args.aug:
    import albumentations
bs = int(args.bs)

use_amp = args.amp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainloader, testloader = get_loader(args)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='resnet18':
    net = resnet18()
elif args.net=="resnet20":
    net = resnet20(args.K)
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='resnet34':
    if not args.quant:
        net = resnet34()
    else:
        net = resnet34(K)
elif args.net=='resnet50':
    net = resnet50()
elif args.net=='resnet101':
    net = resnet101()
elif args.net=="resnext":
    from network.resnext import ResNeXt29_4x64d
    net = ResNeXt29_4x64d()
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = 32,
    patch_size = args.patch,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()
# reduce LR on Plateau
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)    
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs-1)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau"

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device).long()
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(watermark))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

wandb.watch(net)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)
    
    if args.cos:
        scheduler.step()

# writeout wandb
wandb.save("wandb_{}.h5".format(args.net))
