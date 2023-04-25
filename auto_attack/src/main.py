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
        test_ds = CIFAR10(root='./data', train=False, download=True, transform = test_tfm)
        LEN_TEST = len(test_ds)
        classes = 10
    elif config['dataset'] == 'cifar100':
        test_ds = CIFAR100(root='./data', train=False, download=True, transform = test_tfm)
        LEN_TEST = len(test_ds)
        classes = 100
    
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle = False)
    model = ResNet18(classes = classes).to(device)
    model.to(device)
    model.load_state_dict(torch.load(config['model_path']))

    adversary = AutoAttack(model, norm=config['norm'], eps=config['eps'], version='standard')

    model.eval()
    test_acc = 0

    # test_ds.
    # input = rearrange(test_ds.data, 'n s b d -> n d s b')
    # i2 = test_tfm(test_ds.data)
    # x_adv = adversary.run_standard_evaluation(torch.tensor(rearrange(test_ds.data, 'n s b d -> n d s b')), torch.tensor(test_ds.targets), bs=config['batch_size'])

    transformed_data = test_ds.data

    mean = numpy.mean(transformed_data, axis=(0, 1, 2))
    std = numpy.std(transformed_data, axis=(0, 1, 2))

    # Normalize the batch by subtracting the mean and dividing by the standard deviation
    transformed_data = (transformed_data - mean) / std
    adv_complete = adversary.run_standard_evaluation(torch.tensor(rearrange(transformed_data, 'n s b d -> n d s b')).type(torch.FloatTensor),
                torch.tensor(test_ds.targets), bs=config['batch_size'])
    
    torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_norm_{}_plus_{}.pth'.format(
                config['save_dir'], 'aa', 1, 10000, config['eps'], config['norm'], config['dataset']))
    # idx = 0
    # with torch.no_grad():
    #     # for xtest, ytest in zip(x_adv, test_ds.targets):
    #     for xtest, ytest in test_loader:
    #         xtest = xtest.to(device)
    #         ytest_input = ytest.to(device)
    #         test_prob = model(xtest)
    #         test_prob = test_prob.cpu()
    #         test_pred = torch.max(test_prob,1).indices

    #         x_adv = adversary.run_standard_evaluation(xtest, ytest_input, bs=config['batch_size'])
    #         x_adv = x_adv.to(device)
    #         adv_prob = model(x_adv)
    #         adv_prob = adv_prob.cpu()
            
    #         adv_pred = torch.max(adv_prob,1).indices
    #         test_acc += int(torch.sum(test_pred == adv_pred))
    #         idx += 1
    #         print(f"current finished {idx}")
    #     ep_test_acc = test_acc / LEN_TEST
    #     print(f"Test_acc: {ep_test_acc}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    if config['use_wandb']:
        import wandb
        wandb.init(project=config['wandb_project'], config=config)

    attack(config)