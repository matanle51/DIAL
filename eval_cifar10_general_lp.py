import argparse

import foolbox as fb
import torch
import torchvision
from foolbox import PyTorchModel
from torchvision import transforms
from tqdm import tqdm

from models.WideRestNetDANN import WideResNetDANN
from models.wideresnet import WideResNet
from train_utils import eval_test

parser = argparse.ArgumentParser(description='Evaluation code for General Lp norms')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='Batch size for testing (default: 128)')
parser.add_argument('--model-dir', type=str, default='cifar10_model',
                    help='path to model we wish to load')
parser.add_argument('--checkpoint', default='checkpoint_cifar10.pt', type=str,
                    help='path to pretrained model')
parser.add_argument('--norm', default='l2', type=str, help='lp norm to use for attack',
                    choices=['l2', 'l1', 'linf', 'L2DeepFoolAttack', 'LinfDeepFoolAttack'])

args = parser.parse_args([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)


def main():
    model = WideResNetDANN().to(device)
    # model = WideResNet().to(device)
    model = torch.nn.DataParallel(model)

    assert args.checkpoint != ''

    checkpoint = torch.load(f'{args.model_dir}/{args.checkpoint}', map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f'test_accuracy={checkpoint["test_accuracy"]}')
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print('read checkpoint {}'.format(args.checkpoint))

    robust_acc, natural_acc = eval_test(model, test_loader, norm=args.norm)
    print("For attack norm {}, Robust Accuracy: {}, Natural Accuracy: {}".format(args.norm, robust_acc, natural_acc))


def eval_test(model, test_loader, norm):
    model.eval()

    fb_model = PyTorchModel(model, bounds=(0, 1), device=device)

    if norm == 'l2':
        epsilon, num_steps = 0.5, 20
        adversary = fb.attacks.L2PGD(steps=num_steps)
    elif norm == 'l1':
        epsilon, num_steps = 12, 20
        adversary = fb.attacks.L1PGD(steps=num_steps)
    elif norm == 'LinfDeepFoolAttack':
        epsilon, num_steps = 0.02, 50
        adversary = fb.attacks.LinfDeepFoolAttack(steps=num_steps)
    elif norm == 'L2DeepFoolAttack':
        epsilon, num_steps = 0.02, 50
        adversary = fb.attacks.L2DeepFoolAttack(steps=num_steps)
    else:
        raise NotImplementedError(f'Requested norm is not implemented: {norm}')

    print(f'Using adversary with norm={norm}')

    robust_err_total = 0
    natural_err_total = 0
    num_test_samples = len(test_loader.dataset)

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        raw_advs, clipped_advs, success = adversary(fb_model, data, target, epsilons=epsilon)

        out = model(data)
        err_clean = (out.data.max(1)[1] != target.data).float().sum()
        # err_adv = (model(clipped_advs).data.max(1)[1] != target.data).float().sum()
        err_adv = success.float().sum()
        robust_err_total += err_adv
        natural_err_total += err_clean

    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)

    natural_acc = round((num_test_samples - natural_err_total.item()) / num_test_samples * 100, 3)
    print(f'Natural acc total: {natural_acc}')
    robust_acc = round((num_test_samples - robust_err_total.item()) / num_test_samples * 100, 3)
    print(f'Robust acc total: {robust_acc}')

    return robust_acc, natural_acc


if __name__ == '__main__':
    main()
