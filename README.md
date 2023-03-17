Authors: AidenPan, lhxchifan
Date: March, 2023

<img src="[markdownmonstericon.png](https://drive.google.com/file/d/1FRvu7y99OxZG_RRNAKcGMPtwoTgDAA_4/view?usp=sharing)"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

This project is for ECE 740 B03 2023 Winter.

**Detect Adversarial Attacked Images with SOTA Abnormality Detection Algorithms.**

`resnet` folder contains training resnet18 models on CIFAR10 and CIFAR100 dataset. 

Based on trained resnet18 models, `auto_attack` takes in clean test images $I^{test}_{c}$ and train adversarial attacked images $I^{test}_{a}$.

In `abnormaly_detect`, clean training images $I^{train}_{c}$ is fed into the algorithm to train a model which reveals the latent space of clean data. In testing, the model takes in $I^{train}_{c}$ and $I^{test}_{a}$, performs the abnormaly detection to check if it is able to predict the adversarial attacked objects. If is, it proves abnormaly detection algorithm used is capable of predicting AA images.