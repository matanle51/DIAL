import argparse

import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
from autoattack import AutoAttack

from models.WideRestNetDANN import WideResNetDANN
from models.preactresnetDANN import PreActResNetDANN18

parser = argparse.ArgumentParser(description='Test Auto Attack')
parser.add_argument('--epsilon', default=0.031, help='perturbation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size(default: 128)')
parser.add_argument('--norm', type=str, default='Linf', metavar='N', help='Norm to use in Auto-Attack')
parser.add_argument('--version', type=str, default='standard', metavar='N', help='Auto Attack version to run')
parser.add_argument('--data-type', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN'])
parser.add_argument('--model-type', type=str, default='WideResNetDANN', choices=['WideResNetDANN', 'PreActResNetDANN18'])
parser.add_argument('--model-dir', type=str, default='model_cifar10', choices=['model_cifar10', 'model_svhn'])
parser.add_argument('--checkpoint-name', type=str, default='./model.pt')

args = parser.parse_args([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
transform_test = transforms.Compose([transforms.ToTensor()])

if args.data_type == 'SVHN':
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
elif args.data_type == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)


def test_auto_attack():
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, version=args.version, device=device)

    model.eval()

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0).to(device)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).to(device)

    print(f'x_test={len(x_test)}; y_test={len(y_test)}')
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=250)
    pgd_out = model(x_adv, 0)
    err_robust = (pgd_out.data.max(1)[1] != y_test.data).float().sum()
    print(f'err_robust={err_robust}')
    torch.save({'x_adv': x_adv}, f'auto_attack_res.pt')


if __name__ == '__main__':
    # Create model instance
    if args.model_type == 'PreActResNetDANN18':
        model = nn.DataParallel(PreActResNetDANN18()).to(device)
    elif args.model_type == 'WideResNetDANN':
        model = nn.DataParallel(WideResNetDANN()).to(device)
    else:
        raise NotImplementedError(f'Model type not supported: {args.model_type}')
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_auto_attack()
