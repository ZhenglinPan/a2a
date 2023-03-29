import os
from time import time
from tqdm import tqdm
import numpy

import wandb

import torch
from torch.nn import Linear, CrossEntropyLoss, Sequential, MaxPool2d, Flatten, Dropout
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
from util.utils import parse_args, load_config

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
from resnet import ResNet18

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)


def adjust_learning_rate(config, optimizer, epoch):
    """decrease the learning rate"""
    lr = config['learning_rate']
    if epoch >= 20:
        lr = config['learning_rate'] * 0.1
    elif epoch >= 30:
        lr = config['learning_rate'] * 0.05
    elif epoch >= 40:
        lr = config['learning_rate'] * 0.01
    elif epoch >= 50:
        lr = config['learning_rate'] * 0.005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['dataset'] == 'cifar10':
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
    elif config['dataset'] == 'cifar100':
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD

    train_tfm = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_tfm = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    

    
    if config['dataset'] == 'cifar10':
        train_ds = CIFAR10(root='./data', train=True, download=True, transform=train_tfm)
        test_ds = CIFAR10(root='./data', train=False, download=True, transform=test_tfm)
        LEN_TRAIN = len(train_ds)
        LEN_TEST = len(test_ds)
        classes = 10
    elif config['dataset'] == 'cifar100':
        train_ds = CIFAR100(root='./data', train=True, download=True, transform=train_tfm)
        test_ds = CIFAR100(root='./data', train=False, download=True, transform=test_tfm)
        LEN_TRAIN = len(train_ds)
        LEN_TEST = len(test_ds)
        classes = 100
    # model = resnet18(pretrained=False)
    # model.fc = Linear(in_features=512, out_features=10)
    # model.fc = Sequential(Flatten(), Dropout(0.2), Linear(512, classes))

    model = ResNet18(classes = classes)
    print(model)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)


    # for param in model.parameters():
    #   param.requires_grad = False
    # for param in model.fc.parameters():
    #   param.requires_grad = True
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimiser
    # optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.LARS(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # lr_scheduler_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    lr_scheduler_decay = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['learning_rate'], epochs=config['epochs'], 
                                               steps_per_epoch=len(train_loader))
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
                optimizer.zero_grad()
                
                xtrain = xtrain.to(device)
                train_prob = model(xtrain)
                train_prob = train_prob.cpu()
                
                loss = loss_fn(train_prob, ytrain)
                loss.backward()
                optimizer.step()
                
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
        
        lr_scheduler_decay.step()
        # adjust_learning_rate(config, optimizer, epoch)
        
        end = time()
        duration = (end - start) / 60
        
        wandb.log({"train acc": ep_tr_acc, "loss": loss, "test acc": ep_test_acc, "lr": lr_scheduler_decay.get_last_lr()})
        
        print(f"Epoch: {epoch}, Time: {duration}, Loss: {loss}\nTrain_acc: {ep_tr_acc}, Test_acc: {ep_test_acc}, lr: {lr_scheduler_decay.get_last_lr()}")
        
        if (epoch + 1) % config['save_epochs'] == 0:
            torch.save(model.state_dict(), config['save_path'] + "epoch"+str(epoch)+'_train'+str(ep_tr_acc)+'_test'+str(ep_test_acc)+".pt")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    if config['use_wandb']:
        wandb.init(project=config['wandb_project'], config=config)

    path = 'resnet/models/' + config['train_id'] + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    config['save_path'] = path
    train(config)