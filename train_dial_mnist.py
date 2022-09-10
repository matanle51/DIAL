import argparse
import os
import random

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from models.SmallCNNDANN import SmallCNNDANN
from lr_scheduling import adjust_learning_rate
from whitebox_attack import eval_adv_test_whitebox
from train_utils import eval_train, eval_test, train_robust_model, \
    save_best_robust_model

parser = argparse.ArgumentParser(description='DIAL MNIST experiment')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3, help='perturbation')
parser.add_argument('--num-steps', default=40, help='perturb number of steps')
parser.add_argument('--step-size', default=0.01, help='perturb step size')
parser.add_argument('--beta', default=6.0, help='robust loss weight')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='dial/model_mnist', help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N', help='save frequency')
parser.add_argument('--test-attack-freq', '-t', default=1, type=int, metavar='N', help='save frequency')
parser.add_argument('--domain-loss-ratio', '-d', default=0.1, type=float, help='domain regularization in loss function')
parser.add_argument('--distance', default='linf_ce', help='distance')
parser.add_argument('--schedule', default='mnist', help='learning rate schedule')

args = parser.parse_args([])

# folder settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
torch.manual_seed(args.seed)
random.seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup data loader
trainset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


def main():
    # init model, Net() can be also used here for training
    model = SmallCNNDANN().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_robust_acc = 0

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch, init_lr=args.lr, schedule=args.schedule, total_epochs=args.epochs)

        # adversarial training
        train_robust_model(model, device, train_loader, optimizer, epoch,
                           args.log_interval, args.step_size, args.epsilon, args.num_steps,
                           args.beta, args.distance, args.domain_loss_ratio, args.epochs)

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
            natural_acc, robust_acc = eval_adv_test_whitebox(model, device, test_loader,
                                                             len(test_loader.dataset), args.epsilon,
                                                             step_size=args.step_size, num_attack_steps=args.num_steps)
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_best_robust_model(epoch, model, natural_acc, optimizer, robust_acc,
                                       test_accuracy, test_loss, train_accuracy, train_loss,
                                       model_dir=f'{args.model_dir}/best_robust_checkpoint_mnist.pt')


if __name__ == '__main__':
    main()
