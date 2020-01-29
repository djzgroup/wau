import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim.sgd import SGD
from random import choices
from math import floor

 
class WSGD(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, rate=0.1):
        super(WSGD, self).__init__(params, lr, momentum,
                                   dampening, weight_decay, nesterov)
        if rate > 1 or rate <= 0:
            raise ValueError("Invalid rate: {}".format(rate))
        self.rate = rate

    def make_grad_sparse(self, grad):
        if len(grad.size()) == 4:
            mean = torch.mean(grad)
            group = grad.size(1)
            groups = list(range(group))
            update_group = choices(groups, k=floor(group * self.rate))
            # new_grad = torch.full_like(grad, mean).cuda()
            new_grad = torch.zeros_like(grad).cuda()
            new_grad.data[:, update_group] = grad.data[:, update_group]
            return new_grad
        else:
            return grad

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                d_p = self.make_grad_sparse(d_p)
                p.data.add_(-group['lr'], d_p)

        return loss