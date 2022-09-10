# Code based on https://github.com/csdongxian/AWP

from collections import OrderedDict

import torch
import torch.nn.functional as F

EPS = 1E-20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class DialAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(DialAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta, reversal_ratio, domain_pgd_ratio, batch_size):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        class_output_natural, domain_output_natural = self.proxy(inputs_clean, reversal_ratio)
        class_output_adv, domain_output_adv = self.proxy(inputs_adv, reversal_ratio)

        loss_natural = F.cross_entropy(class_output_natural, targets)
        loss_robust = F.kl_div(F.log_softmax(class_output_adv, dim=1),
                               F.softmax(class_output_natural, dim=1),
                               reduction='batchmean')

        natural_domain_label = torch.zeros(batch_size).long().to(device)
        loss_domain_source = F.cross_entropy(domain_output_natural, natural_domain_label)

        adv_domain_label = torch.ones(batch_size).long().to(device)
        loss_domain_target = F.cross_entropy(domain_output_adv, adv_domain_label)

        loss = - 1.0 * (loss_natural + beta * loss_robust + domain_pgd_ratio*(loss_domain_source + loss_domain_target))

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




