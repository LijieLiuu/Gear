import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import CIFAR_MEAN, CIFAR_STD

def get_weak_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def get_train_transform_weak():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def get_train_transform_strong():
    # 简易 strong aug：在 weak 基础上额外颜色扰动与灰度
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

class NpyLabeledDataset(Dataset):
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path)  # (N, 32, 32, 3), uint8
        self.y = np.load(y_path).astype(np.int64)
        self.transform = transform or get_train_transform_weak()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.y[idx])

class NpyUnlabeledDataset(Dataset):
    """返回 weak/strong 两路增强"""
    def __init__(self, X_path, transform_w=None, transform_s=None):
        self.X = np.load(X_path)
        self.tw = transform_w or get_train_transform_weak()
        self.ts = transform_s or get_train_transform_strong()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx])
        return self.tw(img), self.ts(img)

class NpyTestDataset(Dataset):
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path)
        self.y = np.load(y_path).astype(np.int64)
        self.transform = transform or get_weak_transform()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.y[idx])

class ImageFolderUnlabeled(Dataset):
    def __init__(self, root, transform_w=None, transform_s=None):
        self.root = root
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = []
        for r, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(r, f))
        self.paths = sorted(paths)
        self.tw = transform_w or get_train_transform_weak()
        self.ts = transform_s or get_train_transform_strong()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tw(img), self.ts(img)