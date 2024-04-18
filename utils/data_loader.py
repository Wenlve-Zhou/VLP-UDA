from torchvision import datasets, transforms
import torch
import numpy as np
from utils.FixMatch import TransformFixMatch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class toRGB(nn.Module):
    def forward(self, x):
        return x.convert("RGB")

mean,std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

def office_home_partial(dataset):
    partial_classes = sorted(dataset.classes)[:25]
    samples = []
    for (path, label) in dataset.samples:
        class_name = dataset.classes[label]
        if class_name in partial_classes:
            samples.append((path, label))
    dataset.samples = samples
    dataset.classes = partial_classes
    dataset.class_to_idx = [dataset.class_to_idx[c] for c in partial_classes]
    return dataset

def digits_transforms(source,target,data_folder,train):
    if train:
        if source=="svhn" and target=="mnist":
            if source in data_folder:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
            else:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
        elif source == "usps" and target == "mnist":
            if source in data_folder:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(28),
                    transforms.Pad(4),
                    transforms.RandomCrop(28),
                    transforms.RandomRotation(10),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
        elif source == "mnist" and target == "usps":
            if source in data_folder:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
            else:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
        elif source==target:
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    return transform

def load_data(args, data_folder, batch_size, train, num_workers=4, weight_sampler=False, use_fixmatch=False,partial=False,**kwargs):
    if args.datasets == "visda":
        transform = {
            'train': transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)]),
            'test': transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
        }
    else:
        transform = {
            'train': transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize([256, 256]),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)]),
            'test': transforms.Compose(
                [
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
        }

    if not args.datasets == "digits":
        transformation = transform['train' if train else 'test']
    else:
        transformation = digits_transforms(args.src_domain,args.tgt_domain,data_folder,train)
    if use_fixmatch:
        transformation = TransformFixMatch(transformation,args)

    if args.datasets == "digits":
        if "mnist" in data_folder:
            data = datasets.MNIST(root=data_folder, transform=transformation, train=train,download=True)
        elif "usps" in data_folder:
            data = datasets.USPS(root=data_folder,transform=transformation, train=train,download=True)
            data.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        elif "svhn" in data_folder:
            train_ = "train" if train else "test"
            data = datasets.SVHN(root=data_folder,transform=transformation, split=train_,download=True)
            data.classes = ["0","1","2","3","4","5","6","7","8","9"]
    else:
        data = datasets.ImageFolder(root=data_folder, transform=transformation)

    if partial and args.datasets == 'office_home':
        data = office_home_partial(data)

    if weight_sampler:
        weights = get_classes_weight(data)
    else:
        weights = None
    data_loader = get_data_loader(data, batch_size=batch_size,
                                  shuffle=True if train else False,
                                  num_workers=num_workers, weights=weights, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class

def get_classes_weight(dataset):
    y = np.array(dataset.targets)

    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight

def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False,weights=None, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, weights=weights, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        self.dataset = dataset
        self.num_workers = num_workers
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        self.batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            self.dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(self.batch_sampler),
            pin_memory=True,
            prefetch_factor=2
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0