import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cwloss(output, target, confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # Implementation taken from CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    # and FAT https://github.com/zjfheart/Friendly-Adversarial-Training
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


# --- whitebox attack ---
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_attack_steps,
                  step_size,
                  loss_type='ce'):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_attack_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if loss_type == 'ce':
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            elif loss_type == 'cw':
                loss = cwloss(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #     print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader, num_test_samples, epsilon, step_size,
                           num_attack_steps, loss_type='ce'):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    print(f'Using loss type: {loss_type}')

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y,
                                                epsilon=epsilon, num_attack_steps=num_attack_steps,
                                                step_size=step_size, loss_type=loss_type)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)

    natural_acc = (num_test_samples - natural_err_total) / num_test_samples * 100
    print(f'Natural acc total: {natural_acc}')
    robust_acc = (num_test_samples - robust_err_total) / num_test_samples * 100
    print(f'Robust acc total: {robust_acc}')

    return natural_acc, robust_acc
