import os
from time import time
from tqdm import tqdm
import numpy

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms

from utils import parse_args, load_config

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)

def train(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['dataset'] == 'cifar10':
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
    elif config['dataset'] == 'cifar100':
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if config['dataset'] == 'cifar10':
        train_ds = CIFAR10(root='./data', train=True, download=True, transform=tfm)
        test_ds = CIFAR10(root='./data', train=False, download=True, transform=tfm)
        LEN_TRAIN = len(train_ds)
        LEN_TEST = len(test_ds)
        classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    elif config['dataset'] == 'cifar100':
        train_ds = CIFAR100(root='./data', train=True, download=True, transform=tfm)
        test_ds = CIFAR100(root='./data', train=False, download=True, transform=tfm)
        LEN_TRAIN = len(train_ds)
        LEN_TEST = len(test_ds)
        classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle = True)
    model = resnet18(pretrained=True)
    model.fc = Linear(in_features=512, out_features=10)
    model = model.to(device)

    # Optimiser
    optimiser = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)

    # Loss Function
    loss_fn = CrossEntropyLoss()

    for epoch in range(config['epochs']):
        start = time()
        
        tr_acc = 0
        test_acc = 0
        
        # Train
        model.train()
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for xtrain, ytrain in tepoch:
                optimiser.zero_grad()
                
                xtrain = xtrain.to(device)
                train_prob = model(xtrain)
                train_prob = train_prob.cpu()
                
                loss = loss_fn(train_prob, ytrain)
                loss.backward()
                optimiser.step()
                
                # training ends
                
                train_pred = torch.max(train_prob, 1).indices
                tr_acc += int(torch.sum(train_pred == ytrain))
                
            ep_tr_acc = tr_acc / LEN_TRAIN
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            for xtest, ytest in test_loader:
                xtest = xtest.to(device)
                test_prob = model(xtest)
                test_prob = test_prob.cpu()
                
                test_pred = torch.max(test_prob,1).indices
                test_acc += int(torch.sum(test_pred == ytest))
            ep_test_acc = test_acc / LEN_TEST
        
        end = time()
        duration = (end - start) / 60
        
        print(f"Epoch: {epoch}, Time: {duration}, Loss: {loss}\nTrain_acc: {ep_tr_acc}, Test_acc: {ep_test_acc}")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    if config['use_wandb']:
        import wandb
        wandb.init(project=config['wandb_project'], config=config)

    train(config)