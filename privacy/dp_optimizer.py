import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
import os

# Activate CUDA
cuda = True
device = torch.device("cuda:0" if cuda else "cpu")

def _set_seed(secure_seed: int):
    if secure_seed is not None:
        secure_seed = secure_seed
    else:
        secure_seed = int.from_bytes(
            os.urandom(8), byteorder="big", signed=True
        )
    return secure_seed

# Generate secure seed
secure_seed = _set_seed(None)

# Secure generator
_secure_generator = (
    torch.random.manual_seed(secure_seed)
    if device.type == "cpu"
    else torch.cuda.manual_seed(secure_seed)
)

# Generate noise
def _generate_noise(noise_multiplier, max_norm, parameter):
    if noise_multiplier > 0:
        return torch.normal(
            0,
            noise_multiplier * max_norm,
            parameter.grad.shape,
            device=device,
            generator=_secure_generator,
        )
    return 0.0

def create_optimizer(cls):
    class DPOptimizer(cls):
        def __init__(self, max_per_sample_grad_norm, noise_multiplier, minibatch_size, *args, **kwargs):
            super(DPOptimizer, self).__init__(*args, **kwargs)

            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.noise_multiplier = noise_multiplier
            self.minibatch_size = minibatch_size

            # self.param_groups is hidden in the optimizer and **kwargs after calling autoencoder.get_decoder().parameters()
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizer, self).zero_grad()

        def clip_grads_(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.max_per_sample_grad_norm / (total_norm + 1e-8), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()


        def add_noise_(self):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:

                        # Accumulate gradients
                        param.grad.data = accum_grad.clone()

                        # Add noise
                        param.grad.data.add_(_generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param))

                        # Microbatch size is 1.0
                        param.grad.data.mul_(1.0 / self.minibatch_size)

        def step(self, *args, **kwargs):
            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer

# Adam optimizer
AdamDP = create_optimizer(Adam)

