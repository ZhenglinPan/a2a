import torch
import numpy as np
import os
from time import time
from tqdm import tqdm
import numpy
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
from util.utils import parse_args, load_config
from torchvision.datasets import CIFAR10, CIFAR100
from resnet.src.resnet import ResNet18
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from autoattack import AutoAttack
from einops import rearrange
import cv2

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)

def attack(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['dataset'] == 'cifar10':
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
    elif config['dataset'] == 'cifar100':
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD

    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    if config['dataset'] == 'cifar10':
        test_ds = CIFAR10(root='./data', train=False, download=True, transform=test_tfm)
        org_dataset = CIFAR100(root='./data', train=False, download=True)
        LEN_TEST = len(test_ds)
        classes = 10
    elif config['dataset'] == 'cifar100':
        test_ds = CIFAR100(root='./data', train=False, download=True, transform=test_tfm)
        org_dataset = CIFAR100(root='./data', train=False, download=True)
        LEN_TEST = len(test_ds)
        classes = 100
    
    model = ResNet18(classes = classes).to(device)
    model.to(device)
    model.load_state_dict(torch.load(config['model_path']))

    model.eval()
    test_acc = 0

    attacked_data = torch.load(config['data_path'])
    attacked_data_np = attacked_data.numpy()
    attacked_data_np = rearrange(attacked_data_np, 'n s b d -> n b d s')

    transformed_data = test_ds.data
    mean = numpy.mean(transformed_data, axis=(0, 1, 2))
    std = numpy.std(transformed_data, axis=(0, 1, 2))
    attacked_data_np = attacked_data_np * std + mean

    test_ds.data = attacked_data_np.astype(np.uint8)

    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle = False)

    plot_idx = 0
    with torch.no_grad():
        # for xtest, ytest in zip(x_adv, test_ds.targets):
        for idx, (xtest, ytest) in enumerate(test_loader, 0):
            xtest = xtest.to(device)
            test_prob = model(xtest)
            test_prob = test_prob.cpu()
            
            test_pred = torch.max(test_prob,1).indices
            test_acc += int(torch.sum(test_pred == ytest))
            if plot_idx < 10:
                test_pred_v = test_pred.detach().cpu().numpy()[0]
                ytest_v = ytest.detach().cpu().numpy()[0]
                attacked_img = attacked_data_np[idx]
                org_img = org_dataset.data[idx]
                cv2.imwrite("auto_attack/outputs/" + str(plot_idx) + "clean" + str(ytest_v) + ".jpg", org_img)
                cv2.imwrite("auto_attack/outputs/" + str(plot_idx) + "attack" + str(test_pred_v) + ".jpg", attacked_img)
                plot_idx += 1
        ep_test_acc = test_acc / LEN_TEST
        print(f"Test_acc: {ep_test_acc}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    if config['use_wandb']:
        import wandb
        wandb.init(project=config['wandb_project'], config=config)

    attack(config)