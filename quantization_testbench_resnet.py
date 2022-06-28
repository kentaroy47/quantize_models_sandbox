#!/usr/bin/env python
# coding: utf-8

# # Resnet18 CIFAR-10
# 
# Quantize train resnet18 without PACT

# In[1]:


NOQUANT_TRAIN = False
PRETRAIN = False
n_epochs = 200
randamaug = True
bs = 2048
use_amp = True
lr = 1e-3


# In[2]:


import argparse
import os
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import progress_bar

from tqdm.notebook import tqdm
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau


DATA_DIR = "train"
print_freq = 50

# vit imsize
imsize = 32


# In[3]:


# prepare labels
train_df = pd.read_csv("trainLabels.csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])

# 5-fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_df["fold"] = -1
for i, (train_index, test_index) in enumerate(skf.split(train_df.id, train_df.label_enc)):
    train_df.loc[test_index, 'fold'] = i
train_df.head()


# In[4]:


train_df[train_df.fold==1].label.value_counts()


# In[5]:


class cifarDataset(Dataset):
    def __init__(self,
                 df,
                 rand=False,
                 transform=None,
                 test=False
                ):

        self.df = df.reset_index(drop=True)
        self.rand = rand
        self.transform = transform
        self.test = test

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.id
        
        images = cv2.imread(os.path.join(DATA_DIR, str(img_id)+".png"))
        
        # Load labels
        label = row.label_enc
        
        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
              
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        label = label.astype(np.float32)
        #label2 = label2.astype(np.float32)
        return torch.tensor(images), torch.tensor(label),


# In[6]:


import albumentations as A
import albumentations

transforms_train = albumentations.Compose([
    albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=180,p=0.5),
    A.Cutout(num_holes=12, max_h_size=8, max_w_size=8, fill_value=0, p=0.5),
    albumentations.Rotate(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),   
    albumentations.Resize(imsize, imsize, p=1.0), 
])

transforms_val = albumentations.Compose([albumentations.Resize(imsize, imsize, p=1.0),])


# In[7]:


dataset_show = cifarDataset(train_df, transform=transforms_train)
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(3):
    f, axarr = plt.subplots(1,5)
    for p in range(5):
        idx = np.random.randint(0, len(dataset_show))
        img, label = dataset_show[idx]
        img = img.flip(0) #BGR2RGB
        axarr[p].imshow(img.transpose(0,1).transpose(1,2))
        axarr[p].set_title(str(label))
plt.show()


# In[8]:


##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
    
    # Save checkpoint.
    acc = 100.*correct/total
    return test_loss/(batch_idx+1), acc


# In[9]:


# Data
print('==> Preparing data..')
from randomaug import RandAugment
import torchvision
import torchvision.transforms as transforms

size = imsize
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if randamaug:
    N = 2; M = 7;
    transform_train.transforms.insert(0, RandAugment(N, M))

print('==> Preparing dataloader')
trainset = torchvision.datasets.CIFAR10(root='../vision-transformers-cifar10/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../vision-transformers-cifar10/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)


# In[10]:


from network.resnet import resnet20
net = resnet20(abits=8, wbits=8, pact=False, shallow=False)


# In[11]:


net.forward(torch.randn(1,3,32,32))


# # Train with quantization

# from timm.scheduler import CosineLRScheduler
# 
# for shallow in [False]:
#     for pact in [False,]:
#         for k in range(3,9):
#             net = resnet20(k,k,pact,shallow,True)
# 
#             net = net.cuda()
#             mixup = False
# 
#             # Track experiment with wandb
#             import wandb
#             watermark = "resnet18_uniformquantize_k{}".format(k)
#             if pact: watermark+="_pact"
#             if shallow: 
#                 watermark+="_shallow"
#             else:
#                 watermark+="_deep"
#                 
#             # mess with wandb
#             wandb.init(project="quantize_resnet2", name=watermark)
# 
#             # define loss function (criterion) and pptimizer
#             criterion = nn.CrossEntropyLoss().cuda()
# 
#             # optimizer for pact
#             optimizer = torch.optim.SGD(net.parameters(), lr=1e-3,
#                                         momentum=0.9,
#                                         weight_decay=0.0002)
#             optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 
#             scheduler = CosineLRScheduler(optimizer, t_initial=n_epochs, lr_min=1e-6, 
#                                   warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
# 
#             best_prec1 = 0
#             os.makedirs("models", exist_ok=True)
# 
#             for epoch in range(n_epochs):
#                 scheduler.step(epoch)
#                 # train for one epoch
#                 print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
#                 tloss = train(train_loader, model, criterion, optimizer, epoch)         
# 
#                 # evaluate on validation set
#                 prec1, valloss = validate(val_loader, model, criterion)
# 
#                 # wandb
#                 wandb.log({'epoch': epoch, "prec":prec1, "train_loss": tloss, 'val_loss': valloss, "lr": optimizer.param_groups[0]["lr"],})
# 
#                 # remember best prec@1 and save checkpoint
#                 is_best = prec1 > best_prec1
#                 best_prec1 = max(prec1, best_prec1)
# 
#                 print("Best prec1 : ", best_prec1)
#                 if is_best:
#                     torch.save(model.state_dict(), os.path.join(f'models/{watermark}.pth'))

# In[ ]:


from timm.scheduler import CosineLRScheduler
from network.resnet2 import ResNet18
device = "cuda"


if NOQUANT_TRAIN:
    NOQUANT = True
else:
    NOQUANT = False

for shallow in [True, False]:
    for k in range(3,9):
        net = resnet20(k,k,False,shallow,NOQUANT)
        
        if PRETRAIN:
            if shallow:
                checkpoint = torch.load('./models/resnet20_noquant_shallow.pth')
                net.load_state_dict(checkpoint)
            else:
                checkpoint = torch.load('./models/resnet20_noquant_deep.pth')
                net.load_state_dict(checkpoint)

        net = net.cuda()
        mixup = False

        # Track experiment with wandb
        import wandb
        watermark = "resnet20_quant_k{}".format(k)

        if shallow: 
            watermark+="_shallow"
        else:
            watermark+="_deep"
        if not PRETRAIN:
            watermark+="_scratch"

        # mess with wandb
        wandb.init(project="quantize_resnet2", name=watermark)

        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss().cuda()

        # optimizer for pact
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        scheduler = CosineLRScheduler(optimizer, t_initial=n_epochs, lr_min=1e-6, 
                          warmup_t=3, warmup_lr_init=1e-3, warmup_prefix=True)
        
        best_prec1 = 0
        os.makedirs("models", exist_ok=True)

        list_loss = []
        list_acc = []

        net.cuda()
        net = torch.nn.DataParallel(net) # make parallel
        torch.backends.cudnn.benchmark = True
        for epoch in range(n_epochs):
            start = time.time()
            trainloss = train(epoch)
            val_loss, prec1 = test(epoch)

            scheduler.step(epoch-1) # step cosine scheduling

            list_loss.append(val_loss)
            list_acc.append(prec1)

            # Log training..
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "prec": prec1, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            print("Best prec1 : ", best_prec1)
            if is_best:
                torch.save(net.state_dict(), os.path.join(f'models/{watermark}.pth'))


# In[ ]:




