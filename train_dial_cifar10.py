import argparse
import os

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms

from lr_scheduling import adjust_learning_rate
from models.WideRestNetDANN import WideResNetDANN
from train_utils import train_robust_model, eval_train, eval_test, save_best_robust_model
from whitebox_attack import eval_adv_test_whitebox

parser = argparse.ArgumentParser(description='DIAL CIFAR-10 experiment')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=7e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', default=0.031, help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--test-num-steps', default=20, help='perturb number of steps for robust test')
parser.add_argument('--step-size', default=0.007, help='perturb step size')
parser.add_argument('--test-step-size', default=0.003, help='perturb step size for test')
parser.add_argument('--beta', default=8.0, help='robust loss weight')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='dial/model_cifar10', help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', default=1, type=int, metavar='N', help='save frequency')
parser.add_argument('--test-attack-freq', default=1, type=int, metavar='N', help='save frequency')
parser.add_argument('--domain-loss-ratio', default=6, type=float, help='domain regularization in loss function')
parser.add_argument('--distance', default='linf_kl', help='distance')
parser.add_argument('--schedule', default='cifar', help='learning rate schedule')

args = parser.parse_args([])

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

torch.manual_seed(args.seed)

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

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def main():
    # init model, ResNet18() can be also used here for training
    model = WideResNetDANN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_robust_acc = 0

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch, init_lr=args.lr, schedule=args.schedule, total_epochs=args.epochs)

        # adversarial training
        train_robust_model(model, device, train_loader, optimizer, epoch,
                           args.log_interval, args.step_size, args.epsilon, args.num_steps,
                           args.beta, args.distance, args.domain_loss_ratio, args.epochs,
                           awp_args=None, awp_adversary=None)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_accuracy = eval_train(model, device, train_loader)
        test_loss, test_accuracy = eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(args.model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))

        if epoch % args.test_attack_freq == 0:
            natural_acc, robust_acc = eval_adv_test_whitebox(model, device, test_loader, len(test_loader.dataset),
                                                             args.epsilon, step_size=args.test_step_size,
                                                             num_attack_steps=args.test_num_steps)
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_best_robust_model(epoch, model, natural_acc, optimizer, robust_acc,
                                       test_accuracy, test_loss, train_accuracy, train_loss,
                                       model_dir=f'{args.model_dir}/best_robust_checkpoint_cifar10.pt')


if __name__ == '__main__':
    main()
