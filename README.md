*Authors: AidenPan, lhxchifan*

*Date: March, 2023*

![A2A](https://github.com/ZhenglinPan/a2a/blob/master/others/readme_A2A.png)

This project is for ECE 740 B03 2023 Winter.

**Detect Adversarial Attacked Images with SOTA Abnormality Detection Algorithms.**

`resnet` folder contains training resnet18 models on CIFAR10 and CIFAR100 dataset. 

Based on trained resnet18 models, `auto_attack` takes in clean test images $I^{test}_{c}$ and train adversarial attacked images $I^{test}_{a}$.

In `abnormaly_detect`, clean training images $I^{train}_{c}$ is fed into the algorithm to train a model which reveals the latent space of clean data. In testing, the model takes in $I^{train}_{c}$ and $I^{test}_{a}$, performs the abnormaly detection to check if it is able to predict the adversarial attacked objects. If is, it proves abnormaly detection algorithm used is capable of predicting AA images.

## Environment
```
conda create --name env_name python=3.8
source activate env_name
# install torch by yourself, there is no torch in the requirements.txt
pip install -r requirements.txt
```
## ResNet Training
```
python resnet/src/main.py --config configs/resnet/config.yaml
```  
current config is for CIFAR100 training, if you want to train a model for CIFAR10, modify the config.yaml by yourself.
## Auto-Attack
### CIFAR10
1. download pretrained model from https://drive.google.com/file/d/1LroOpL2KsPTst0K6hSP0UNeYoD816-CH/view?usp=sharing  
2. modify config file configs/auto_attack/config.yaml for `model_path`, `norm`, `eps`  
3. run command  
```
python auto_attack/src/main.py --config configs/auto_attack/config.yaml
```
### CIFAR100
1. download pretrained model from https://drive.google.com/file/d/1iRdgVYVPbbxi4pctokImXJ7E2J9O96H0/view?usp=sharing  
2. modify config file configs/auto_attack/config.yaml for `model_path`, `norm`, `eps`  
3. run command  
```
python auto_attack/src/main.py --config configs/auto_attack/config.yaml
```
## Anomaly Detection
1. download attacked data  
CIFAR10 Linf https://drive.google.com/file/d/1bNFBZRdAl4uNPMrdlU9QC7Vr_ktaxLK5/view?usp=sharing  
CIFAR100 Linf https://drive.google.com/file/d/1lQLOkVKrjWl8YHX1d-VhYqWtHIn3Hku_/view?usp=sharing  
CIFAR10 L2 https://drive.google.com/file/d/1PPCYmUYMjmnNtbC2VWu4Xz9Z5fVycYTt/view?usp=sharing  
CIFAR100 L2 https://drive.google.com/file/d/1uOJBBed_eQZQV9R4zbp1O9xC6BHvqY1W/view?usp=sharing  
replace parameter `attacked_data_file` with the file you want to test  
### MSCL
Code is based on repo https://github.com/talreiss/Mean-Shifted-Anomaly-Detection  
```
python abnomaly_detect/src/msad/main.py --dataset=cifar10 --label=1 --backbone=152 --attacked_data_file auto_attack/outputs/aa_1_individual_1_10000_eps_0.03100_plus_cifar10.pth 
```
### PANDA
Code is based on repo https://github.com/talreiss/PANDA  
```
python abnomaly_detect/src/panda/main.py --dataset=cifar10 --label=1 --resnet_type=152 --ewc --diag_path=abnomaly_detect/data/fisher_diagonal.pth --attacked_data_file auto_attack/outputs/aa_1_individual_1_10000_eps_0.03100_plus_cifar10.pth
``` 
### FITYMI
Code is based on repo https://github.com/rohban-lab/fitymi  

train model
```
python abnomaly_detect/src/FITYMI/main.py --dataset cifar10 --label 0 --output_dir results_cifar10 --normal_data_path data --gen_data_path  abnomaly_detect/data/cifar10_training_gen_data.npy --pretrained_path abnomaly_detect/models/ViT-B_16.npz --nnd --download_dataset --all_label --attacked_data_file auto_attack/outputs/aa_1_individual_1_10000_eps_0.03100_plus_cifar10.pth
```

use pre-trained model
1. download model  
CIFAR10 https://drive.google.com/drive/folders/1hA1k2k81lXjFt0MwMjGAXgA71a1qGAe9?usp=sharing  
CIFAR100 https://drive.google.com/drive/folders/1PpShrjJjy8Gcjc3jcP_9rOU69cht1G99?usp=sharing  
2. put them under root folder.  
```
python abnomaly_detect/src/FITYMI/main.py --dataset cifar10 --label 0 --output_dir results_cifar10 --normal_data_path data --gen_data_path  abnomaly_detect/data/cifar10_training_gen_data.npy --pretrained_path abnomaly_detect/models/ViT-B_16.npz --nnd --download_dataset --all_label --load_model --attacked_data_file auto_attack/outputs/aa_1_individual_1_10000_eps_0.03100_plus_cifar10.pth
```