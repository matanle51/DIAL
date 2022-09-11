# Domain Invariant Adversarial Training (DIAL)
This repository provides code for training and evaluation of Domain Invariant Adversarial Learning (DIAL) method.


## Preferred Prerequisites:

- python = 3.9.2
- torch = 1.8.1
- torchvision = 0.9.1

## Trained models
Trained models for CIFAR-10, CIFAR-100, and SVHN are available [here](https://drive.google.com/drive/folders/1osCczvaoA88WLpAkUrHBn4yxX2sEzbXH?usp=sharing)

## How to run DIAL:

To run DIAL on CIFAR-10 dataset with WRN-34-10
```
python train_dial_cifar10.py
```
To run DIAL on CIFAR-100 dataset with PreAct ResNet-18
```
python train_dial_cifar100.py
```
To run DIAL on SVHN dataset with PreAct ResNet-18
```
python train_dial_svhn.py
```
To run DIAL on MNIST dataset with SmallCNN
```
python train_dial_mnist.py
```

White-box evaluation can be done by calling:
```
from whitebox_attack import eval_adv_test_whitebox
natural_acc, robust_acc = eval_adv_test_whitebox(model, device, test_loader, num_test_samples, 
                                                 epsilon, step_size, num_attack_steps)
```
with the desired parameters

Black-box evaluation can be done by calling:
```
from blackbox_attack import eval_adv_test_blackbox
natural_acc, robust_acc = eval_adv_test_blackbox(model_target, model_source, device, test_loader, num_test_samples,
                                                 epsilon, step_size, num_attack_steps)
```
with the desired parameters

## Results:
### White-box and Auto-Attack (AA) [6] results on CIFAR-10 using different L_inf adversaries
| Defense Model      | Natural         | PGD-20         | PGD-100 | CW | AA     |
| ------------------ |---------------- | -------------- | ------- | ------ |-----   |
| TRADES       [1]   | 84.92           | 56.60          | 55.56   | 54.20  |  53.08 |
| MART         [2]   | 83.62           | 58.12          | 56.48   | 53.09  |  51.10 |
| Madry et al. [3]   | 85.10           | 56.28          | 54.46   | 53.99  |  51.52 |
| Song et al   [4]   | 76.91           | 43.27          | 41.13   | 41.01  |  40.08 |
| **DIAL_CE**        | **89.59**          | 54.31         | 51.67   | 52.04  |  49.85 |
| **DIAL_KL**        | 85.25           | **58.43**      | **56.80**   |  **55.00** | **53.75**|

Incorporating AWP [5] into the training process:

| Defense Model      | Natural         | PGD-20         | PGD-100 | CW | AA     |
| ------------------ |---------------- | -------------- | ------- | ------ | ----   |
| **DIAL-AWP**       | **85.91**           | **61.10**          |  **59.86**  | **57.67**  | **56.78**  |
| DIAL-TRADES [5]    | 85.36           | 59.27          |  59.12  | 57.07  | 56.17  |


### Black-box attack results on CIFAR-10 using different L_inf adversaries
| Defense Model      | Natural         | PGD-20         | PGD-100 | CW_inf |
| ------------------ |---------------- | -------------- | ------- | ------ |
| TRADES       [1]   | 84.92           |  84.08         |   83.89 |  83.91 |
| MART         [2]   | 83.62           |  82.82         |   82.52 |  82.80 |
| Madry et al. [3]   | 85.10           |  84.22         |   84.14 |  83.92 |
| Song et al   [4]   | 76.91           |  75.59         |   75.37 |  75.35 | 
| **DIAL_KL**        | 85.25           |  84.30         | 84.18   |  84.05 | 
| **DIAL_CE**        | **89.59**          |  **88.60**     | **88.39**| **88.44** | 

Incorporating AWP [5] into the training process:

| Defense Model      | Natural         | PGD-20         | PGD-100 | CW_inf |
| ------------------ |---------------- | -------------- | ------- | ------ |
| **DIAL-AWP**       | **85.91**       | **85.13**      | **84.93**   | **85.03** |
| DIAL-TRADES [5]    | 85.36           | 84.58          | 84.58   | 84.59  |

Additional results on transfer learning, unforeseen corruptions, unforeseen adversaries, as well as ablation studies
can be found in the main paper.

## References:
[1] TRADES: https://github.com/yaodongyu/TRADES/

[2] MART: https://github.com/YisenWang/MART

[3] Madry et al.: https://github.com/MadryLab/cifar10_challenge

[4] Song et al.: https://github.com/corenel/pytorch-atda

[5] AWP: https://github.com/csdongxian/AWP

[6] Auto-Attack: https://github.com/fra31/auto-attack
