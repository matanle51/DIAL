import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dial_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.003,
              epsilon=0.031,
              perturb_steps=10,
              beta=1.0,
              distance='linf_kl',
              reversal_ratio=1.0,
              domain_ratio=1.0,
              awp_args=None,
              awp_adversary=None,
              epoch=None):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'linf_kl':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'linf_ce':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Switch model to train mode
    model.train()

    # calculate adversarial weight perturbation
    awp = apply_awp(awp_adversary, awp_args, batch_size, beta, domain_ratio, epoch, reversal_ratio, x_adv, x_natural, y)

    # zero gradient
    optimizer.zero_grad()

    # calculate natural images loss
    class_output_natural, domain_output_natural = model(x_natural, reversal_ratio)
    natural_domain_label = torch.zeros(batch_size).long().to(device)
    loss_natural = F.cross_entropy(class_output_natural, y) + domain_ratio * F.cross_entropy(domain_output_natural,
                                                                                             natural_domain_label)

    # Calculate adversarial images output
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    class_output_adv, domain_output_adv = model(x_adv, reversal_ratio)
    adv_domain_label = torch.ones(batch_size).long().to(device)

    # Calculate adversarial loss with respect to distance metric
    if distance == 'linf_ce':
        loss_adv = F.cross_entropy(class_output_adv, y) * beta + \
                   domain_ratio * F.cross_entropy(domain_output_adv, adv_domain_label)
    elif distance == 'linf_kl':
        loss_adv = (1.0 / batch_size) * criterion_kl(F.log_softmax(class_output_adv, dim=1),
                                                     F.softmax(class_output_natural, dim=1)) * beta + \
                   domain_ratio * F.cross_entropy(domain_output_adv, adv_domain_label)
    else:
        raise NotImplemented(f'Distance metric is not implemented: {distance}')

    loss = loss_natural + loss_adv
    return loss, awp


def apply_awp(awp_adversary, awp_args, batch_size, beta, domain_pgd_ratio, epoch, reversal_ratio, x_adv, x_natural, y):
    if awp_args is not None and awp_args['use_awp'] and epoch >= awp_args['awp_warmup']:
        if awp_args['print_start']:
            print('Starting awp training')
            awp_args['print_start'] = False
        awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                     inputs_clean=x_natural,
                                     targets=y,
                                     beta=beta,
                                     reversal_ratio=reversal_ratio,
                                     domain_pgd_ratio=domain_pgd_ratio,
                                     batch_size=batch_size)
        awp_adversary.perturb(awp)
    else:
        awp = None
    return awp
