from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch

def generate_loaders(dataset_name, val_set_size, train_bs, val_bs, test_bs, n_workers, augment):
    if dataset_name == 'cifar10':
        if augment:
            mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])  
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
          
        d_fun = datasets.CIFAR10
        n_classes = 10
    elif dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])      
        d_fun = datasets.MNIST
        n_classes = 10
    
    addr = './data/' + dataset_name
    train_dataset = d_fun(addr, train=True, download=True, transform=train_transform)
    val_dataset = d_fun(addr, train=True, download=True, transform=test_transform)

    label_dict = {}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        if label not in label_dict:
            label_dict[label] = [idx]
        else:
            label_dict[label].append(idx)

    train_indices = []
    val_indices = []
    for label, idxs in label_dict.iteritems():
        np.random.shuffle(idxs)
        # possible bug here
        train_indices += idxs[(val_set_size/n_classes):]
        val_indices += idxs[:(val_set_size/n_classes)]

    test_dataset = d_fun(addr, train=False, download=True, transform=test_transform)
    assert val_set_size < len(train_dataset)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=train_sampler,
        batch_size=train_bs, num_workers=n_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, sampler=valid_sampler,
        batch_size=val_bs, num_workers=n_workers, pin_memory=True)    
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
        batch_size=test_bs, num_workers=n_workers, pin_memory=True) 

    return train_loader, val_loader, test_loader
