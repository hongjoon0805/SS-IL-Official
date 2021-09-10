from torchvision import datasets, transforms
import torch
import numpy as np

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

def make_dataset(file_path, classes, test_img_per_class = 50):
    data = datasets.ImageFolder(file_path)
    loader = data.loader

    img_cnt = np.zeros(20000)
    img_mask = np.zeros(20000)
    target_map = np.zeros(20000)
    
    # Count the number of images per class
    for i in range(len(data.imgs)):
        path, target = data.imgs[i]
        img_cnt[target] += 1

    idx = classes
    img_cnt_args = np.flip(np.argsort(img_cnt), axis=0)[:idx]
    img_mask[img_cnt_args] = 1
    target_map[img_cnt_args] = np.arange(idx)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    img_cnt = np.zeros(20000)
    for i in range(len(data.imgs)):
        path, target = data.imgs[i]
        if img_mask[target] == 0:
            continue
        if img_cnt[target] < test_img_per_class:
            test_data.append(path)
            test_labels.append(int(target_map[target]))
        else:
            train_data.append(path)
            train_labels.append(int(target_map[target]))

        img_cnt[target] += 1


    train_data = np.stack(train_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    
    return train_data, train_labels, test_data, test_labels, loader

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        
    def shuffle_data(self, shuffle_idx):
        self.train_labels = shuffle_idx[self.train_labels]
        self.test_labels = shuffle_idx[self.test_labels]
        train_sort_index = np.argsort(self.train_labels)
        test_sort_index = np.argsort(self.test_labels)

        self.train_labels = self.train_labels[train_sort_index]
        self.test_labels = self.test_labels[test_sort_index]
        self.train_data = self.train_data[train_sort_index]
        self.test_data = self.test_data[test_sort_index]

        return

class Imagenet(Dataset):
    def __init__(self):
        super().__init__(1000, "Imagenet")
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        
        train_data = datasets.ImageFolder("../dat/Imagenet/train", transform=self.train_transform)
        test_data = datasets.ImageFolder("../dat/Imagenet/val", transform=self.test_transform)
        self.loader = train_data.loader
        
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        
        for i in range(len(train_data.imgs)):
            path, target = train_data.imgs[i]
            self.train_data.append(path)
            self.train_labels.append(target)
            
        for i in range(len(test_data.imgs)):
            path, target = test_data.imgs[i]
            self.test_data.append(path)
            self.test_labels.append(target)
        
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        
        print(len(self.train_data))
            
        
class Google_Landmark_v2_1K(Dataset):
    # First, download google landmark dataset v2
    # Link: https://github.com/cvdfoundation/google-landmark
    # Second, Refine the directory so that it is usable for supervised learning
    
    def __init__(self):
        super().__init__(1000, "Google_Landmark_v2_1K")
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        
        train_data, train_labels, test_data, test_labels, loader = make_dataset("../dat/google-landmark-v2/train_10k", self.classes)
        
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.loader = loader
        
class Google_Landmark_v2_10K(Dataset):
    def __init__(self):
        super().__init__(10000, "Google_Landmark_v2_10K")
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        
        train_data, train_labels, test_data, test_labels, loader = make_dataset("../dat/google-landmark-v2/train_10k", self.classes, test_img_per_class=10)
        
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.loader = loader
        
