import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import pandas as pd
import cv2
import os
import numpy as np
from augs.randomaug import RandAugment
from PIL import Image

DATA_DIR = "train"

logger = logging.getLogger(__name__)

class cifarDataset(Dataset):
    def __init__(self,
                 df,
                 rand=False,
                 transform=None,
                 test=False,
                 aug = False
                ):

        self.df = df.reset_index(drop=True)
        self.rand = rand
        self.transform = transform
        self.test = test
        self.aug = aug

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.aug != True:
            row = self.df.iloc[index]
            img_id = row.id
            images = cv2.imread(os.path.join(DATA_DIR, str(img_id)+".png"))        
            # Load labels
            label = row.label_enc        
            # aug
            if self.transform is not None:
                images = self.transform(image=images)['image']

            # Mixup part
            """
            rd = np.random.rand()
            label2 = label
            gamma = np.array(np.ones(1)).astype(np.float32)[0]
            if mixup and rd < 0.5 and self.test is False:
                mix_idx = np.random.random_integers(0, len(self.df))
                row2 = self.df.iloc[mix_idx]
                img_id2 = row2.id
                images2 = cv2.imread(os.path.join(DATA_DIR, str(img_id2)+".png"))

                if self.transform is not None:
                    images2 = self.transform(image=images2)['image']

                # blend image
                gamma = np.array(np.random.beta(1,1)).astype(np.float32)
                images = ((images*gamma + images2*(1-gamma))).astype(np.uint8)
                # blend labels
                label2 = row2.label_enc
            """

            images = images.astype(np.float32)
            images /= 255
            images = images.transpose(2, 0, 1)

            label = label.astype(np.float32)
            #label2 = label2.astype(np.float32)
            return torch.tensor(images), torch.tensor(label), 
        else:
            # use scikit for torchvision
            row = self.df.iloc[index]
            img_id = row.id
            image = Image.open(os.path.join(DATA_DIR, str(img_id)+".png"))

            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(row.label_enc)
        
def get_loader(args, inference=False):
    try:
        aug = args.aug
    except:
        if inference:
            aug = True
        else:
            aug = False
        
    train_df = pd.read_csv("trainLabels.csv")
    train_df.head()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df['label_enc'] = le.fit_transform(train_df['label'])
    train_df.head()

    # 5-flod
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_df["fold"] = -1
    for i, (train_index, test_index) in enumerate(skf.split(train_df.id, train_df.label_enc)):
        train_df.loc[test_index, 'fold'] = i

    import albumentations as A
    import albumentations

    imsize = 32
    if not aug and not inference:
        transform_train = albumentations.Compose([
            albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=30,p=0.5),
            #A.OneOf([
            #    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
            #                         val_shift_limit=0.2, p=0.5),
            #    A.RandomBrightnessContrast(brightness_limit=0.2, 
            #                               contrast_limit=0.2, p=0.5),
            #],p=0.9),
            A.Cutout(num_holes=16, max_h_size=4, max_w_size=4, fill_value=0, p=0.5),
            #albumentations.Rotate(p=0.5),
            #albumentations.Transpose(p=0.5),
            #albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),   
            albumentations.Resize(imsize, imsize, p=1.0), 
        ])
        transform_test = albumentations.Compose([albumentations.Resize(imsize, imsize, p=1.0)])
    else:
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
        
        # add random augment
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))
    
    try:
        if args.dataset=="cifar100":
            print("using cifar100")
            trainset = datasets.CIFAR100(root="./data",
                                         train=True,
                                         download=True,
                                         transform=transform_train)
            testset = datasets.CIFAR100(root="./data",
                                        train=False,
                                        download=True,
                                        transform=transform_test) if args.local_rank in [-1, 0] else None    
        else:
            print("using cifar10")
            trainset = cifarDataset(train_df[train_df.fold!=0], transform=transform_train, aug=aug)
            testset = cifarDataset(train_df[train_df.fold==0], transform=transform_test, test=True, aug=aug)
            """
            trainset = datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transform_train)
            testset = datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transform_test) if args.local_rank in [-1, 0] else None
           """
    except:
        trainset = cifarDataset(train_df[train_df.fold!=0], transform=transform_train, aug=aug)
        testset = cifarDataset(train_df[train_df.fold==0], transform=transform_test, test=True, aug=aug)
    
    try:
        bs = args.bs
    except:
        bs = 256
    
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=int(bs),
                              num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=int(bs)*2,
                             num_workers=8,
                             pin_memory=True)

    return train_loader, test_loader
