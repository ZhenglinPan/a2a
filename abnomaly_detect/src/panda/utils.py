import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
from einops import rearrange
import numpy
from sklearn.neighbors import NearestNeighbors

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def knn_accuracy(train_set, test_set, labels, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    nn = NearestNeighbors(n_neighbors=n_neighbours)
    nn.fit(train_set)
    result = []
    for idx, test_case in enumerate(test_set):
        input = np.expand_dims(test_case, axis=0)
        distance, indices = nn.kneighbors(input)
        threshold = distance.max() * 0.5
        result.append(1 if distance.min() <= threshold else 0)
    result = np.array(result)
    matches = np.count_nonzero(np.array_equal(result, labels))
    return matches / len(labels)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader


def get_all_loaders(dataset, batch_size, attacked_data_file):
    if dataset in ['cifar10', 'cifar100']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

            attacked_data = torch.load(attacked_data_file)
            attacked_data_np = attacked_data.numpy()
            attacked_data_np = rearrange(attacked_data_np, 'n s b d -> n b d s')

            transformed_data = testset.data
            mean = numpy.mean(transformed_data, axis=(0, 1, 2))
            std = numpy.std(transformed_data, axis=(0, 1, 2))
            attacked_data_np = attacked_data_np * std + mean
            testset.data = numpy.concatenate((testset.data, attacked_data_np.astype(np.uint8)), axis = 0)
            testset.targets = [0 for t in range(10000)] + [1 for t in range(10000)]
            trainset.targets = [1 for i in range(50000)]
        elif dataset == "cifar100":
            ds = torchvision.datasets.CIFAR100
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)


            attacked_data = torch.load(attacked_data_file)
            attacked_data_np = attacked_data.numpy()
            attacked_data_np = rearrange(attacked_data_np, 'n s b d -> n b d s')

            transformed_data = testset.data
            mean = numpy.mean(transformed_data, axis=(0, 1, 2))
            std = numpy.std(transformed_data, axis=(0, 1, 2))
            attacked_data_np = attacked_data_np * std + mean
            testset.data = numpy.concatenate((testset.data, attacked_data_np.astype(np.uint8)), axis = 0)
            testset.targets = [0 for t in range(10000)] + [1 for t in range(10000)]
            trainset.targets = [1 for i in range(50000)]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()


def get_loaders(dataset, label_class, batch_size):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


