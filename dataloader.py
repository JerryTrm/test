import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

mean= [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean= mean, std= std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean= mean, std= std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean= mean, std= std)
    ]),
}

train_dir= ''
test_dir= '/hdd1t/users/vietnh41/html/datasets/test/phong_may/'

Train = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
Test = datasets.ImageFolder(test_dir, transform = data_transforms['test'])

load_transfer = {
    'trainloader': torch.utils.data.DataLoader(Train, batch_size= 8, shuffle= True, num_workers= 4),
    'testloader': torch.utils.data.DataLoader(Test, batch_size= 8, shuffle= False, num_workers= 4)
}


class_names= Train.classes