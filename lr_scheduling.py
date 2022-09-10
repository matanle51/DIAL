def adjust_learning_rate(optimizer, epoch, init_lr, schedule, total_epochs):
    """decrease the learning rate"""
    lr = init_lr
    if schedule == 'cifar':
        if epoch >= 0.75 * total_epochs:
            lr = init_lr * 0.1
        if epoch >= 0.9 * total_epochs:
            lr = init_lr * 0.01
        if epoch >= total_epochs:
            lr = init_lr * 0.001
    elif schedule in ('mnist', 'svhn'):
        if epoch >= 0.55 * total_epochs:
            lr = init_lr * 0.1
        if epoch >= 0.75 * total_epochs:
            lr = init_lr * 0.01
        if epoch >= 0.9 * total_epochs:
            lr = init_lr * 0.001
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Update learning rate to: {lr}')