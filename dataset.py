import glob
import random
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from auto_augment import rand_augment_transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 error

class CustomDownSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.labels = np.array(dataset.label)
        self.num_classes = len(np.unique(self.labels))

        num_each_class = []
        for i in range(self.num_classes):
            num_class_i = (self.labels==i).sum()
            num_each_class.append(num_class_i)
        self.num_each_class = np.array(num_each_class).min()

        self.num_samples = self.num_each_class * self.num_classes

    def __iter__(self):
        idxs = []
        for i in range(self.num_classes):
            idxs_i = np.where(self.labels==i)[0]
            idxs_i = np.random.choice(idxs_i, self.num_each_class, replace=False)
            idxs += idxs_i.tolist()
        
        random.shuffle(idxs)
        
        return (idx for idx in idxs)

    def __len__(self):
        return self.num_samples

# RAF-DB Dataset
class RafDataset(Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx

# AffectNet Dataset
class AffectDataset(torchvision.datasets.ImageFolder):
    def __init__(self, AffectNet_path, phase, transform=None):
        if phase == 'train':
            root_dir = os.path.join(AffectNet_path, 'train')
        else:
            root_dir = os.path.join(AffectNet_path, 'val')
        super().__init__(root=root_dir, transform=transform)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)  # 默认用的是 PIL.Image.open
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

# SFEW Dataset
class SFEWDataset(torchvision.datasets.ImageFolder):
    def __init__(self, sfew_path, phase, transform=None):
        if phase == 'train':
            super().__init__(os.path.join(sfew_path, 'Train'), transform=transform)
        else:
            super().__init__(os.path.join(sfew_path, 'Val'), transform=transform)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx
    
class FER2013Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, fer2013_path, phase, transform=None):
        if phase == 'train':
            root_dir = os.path.join(fer2013_path, 'train')
        else:
            root_dir = os.path.join(fer2013_path, 'val')
        super().__init__(root=root_dir, transform=transform)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)  # 默认用的是 PIL.Image.open
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

class FERPlusDataset(torchvision.datasets.ImageFolder):
    def __init__(self, ferplus_path, phase, transform=None):
        if phase == 'train':
            root_dir = os.path.join(ferplus_path, 'train')
        else:
            root_dir = os.path.join(ferplus_path, 'val')
        super().__init__(root=root_dir, transform=transform)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)  # 默认用的是 PIL.Image.open
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx
    
class NormalizeCrops:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        return torch.stack([transforms.Normalize(mean=self.mean, std=self.std)(t) for t in tensors])
    
class ApplyRandAugment:
    def __init__(self):
        self.transform_fn = rand_augment_transform(
            config_str='rand-m3-n5-mstd0.5',
            hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
        )

    def __call__(self, crops):
        return torch.stack([transforms.ToTensor()(self.transform_fn(crop)) for crop in crops])


class ApplyHorizontalFlip:
    def __call__(self, crops):
        return torch.stack([transforms.RandomHorizontalFlip()(crop) for crop in crops])

class ApplyRandomErasing:
    def __call__(self, tensors):
        return torch.stack([transforms.RandomErasing(scale=(0.02, 0.25))(t) for t in tensors])

class ToTensorCrops:
    def __call__(self, crops):
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

class NormalizeCrops:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        return torch.stack([transforms.Normalize(mean=self.mean, std=self.std)(t) for t in tensors])

def get_dataloaders(dataset='AffectNet', data_path='./datasets/AffectNet', batch_size=64, num_workers=0, num_samples=30000, pin_memory=False):
    # transforms 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    if dataset in ['raf']:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            rand_augment_transform(config_str='rand-m5-n3-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(scale=(0.02, 0.25)),
            ])
        data_transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
    elif dataset in ['AffectNet', 'FER-2013', 'FERPlus']: 
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(ApplyRandAugment()),
            transforms.Lambda(ApplyHorizontalFlip()),
            transforms.Lambda(NormalizeCrops(mean, std)),
            transforms.Lambda(ApplyRandomErasing()),
        ])


        data_transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            ToTensorCrops(),
            NormalizeCrops(mean, std),
        ])

    # datasets
    if dataset == 'raf':
        dataset = RafDataset
    elif dataset == 'AffectNet':
        dataset = AffectDataset
    elif dataset == 'sfew':
        dataset = SFEWDataset
    elif dataset == 'FER-2013':
        dataset = FER2013Dataset
    elif dataset == 'FERPlus':
        dataset = FERPlusDataset

    train_dataset = dataset(
        data_path,
        phase='train',
        transform=data_transforms
    )
    val_dataset = dataset(
        data_path,
        phase='test',
        transform=data_transforms_val
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, 
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=True
    )

    return train_loader, val_loader
