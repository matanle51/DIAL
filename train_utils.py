import math

import torch
import torch.nn.functional as F

from dial import dial_loss


def train_robust_model(model, device, train_loader, optimizer, epoch,
                       log_interval, step_size, epsilon, num_steps, beta,
                       distance, domain_ratio, n_epochs, awp_args=None, awp_adversary=None):
    agg_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Adjust reversal ratio
        reversal_ratio = adjust_alpha(batch_idx, epoch, len(train_loader.dataset), n_epochs)

        # calculate robust loss
        loss, awp = dial_loss(model=model,
                              x_natural=data,
                              y=target,
                              optimizer=optimizer,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=num_steps,
                              beta=beta,
                              distance=distance,
                              reversal_ratio=reversal_ratio,
                              domain_ratio=domain_ratio,
                              awp_args=awp_args,
                              awp_adversary=awp_adversary,
                              epoch=epoch)

        loss.backward()
        optimizer.step()

        # awp integration if requested
        if awp is not None and awp_args is not None and awp_args['use_awp'] and epoch >= awp_args['awp_warmup']:
            awp_adversary.restore(awp)

        agg_loss += loss.item()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\treversal_ratio: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), agg_loss / (batch_idx + 1), round(reversal_ratio, 7)))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def save_best_robust_model(epoch, model, natural_acc, optimizer, robust_acc, test_accuracy, test_loss, train_accuracy,
                           train_loss, model_dir):
    torch.save({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_dir)
    print(f'Best model was found with: robust_acc={robust_acc}; natural_acc={natural_acc}')


def adjust_alpha(i, epoch, dataset_len, nepochs):
    p = float(i + epoch * dataset_len) / nepochs / dataset_len
    o = -10
    alpha = 2. / (1. + math.exp(o * p)) - 1
    return alpha
