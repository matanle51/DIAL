import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
    out = model_target(X)

    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_blackbox(model_target, model_source, device, test_loader, num_test_samples,
                           epsilon, step_size, num_attack_steps):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y,
                                                epsilon, num_attack_steps, step_size)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)

    natural_acc = (num_test_samples - natural_err_total) / num_test_samples * 100
    print(f'Natural acc total: {natural_acc}')
    robust_acc = (num_test_samples - robust_err_total) / num_test_samples * 100
    print(f'Robust acc total: {robust_acc}')

    return natural_acc, robust_acc
