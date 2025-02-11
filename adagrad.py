import torch
from torch.optim.optimizer import Optimizer

class AdaGrad(Optimizer):

    def __init__(self, params, lr=0.01, eps=1e-10):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)
        torch.manual_seed(1234567)

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:

                # Skip Parameters with None gradients
                if p.grad is None:
                    continue

                # Retreive the parameter's current state and gradient
                grad = p.grad.data
                param_state = self.state[p]

                if 'sum' not in param_state:
                    param_state['sum'] = torch.zeros_like(p.data)

                # Update the current sum of gradients
                param_state['sum'].addcmul_(grad, grad)

                # Update the parameter using AdaGrad update rule:
                # X = X - (alpha / sqr(G) + eps) * grad
                sum_sqrt_grad = param_state['sum'].sqrt().add_(group['eps'])
                scaled_lr = torch.div(group['lr'], sum_sqrt_grad)
                p.data.addcmul_(grad, -scaled_lr)
        
        return loss