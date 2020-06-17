import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
import os
import numpy as np

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
        def __init__(self, max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs):
            super(DPOptimizer, self).__init__(*args, **kwargs)

            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.noise_multiplier = noise_multiplier
            self.batch_size = batch_size

            # self.param_groups is hidden in the optimizer and **kwargs after calling autoencoder.parameters()
            for group in self.param_groups:
                group['aggregate_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        # May not be necessary as we can use optimizer.zero_grad() in the loop.
        def zero_microbatch_grad(self):
            super(DPOptimizer, self).zero_grad()

        def clip_grads_(self):

            # Clip gradients in-place
            params = self.param_groups[0]['params']
            clip_grad_norm_(params, max_norm=self.max_per_sample_grad_norm, norm_type=2)

            # Accumulate gradients
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data)

            ### Original Implementation below ###
            # total_norm = 0.
            # for group in self.param_groups:
            #     total_norm_params = np.sum([param.grad.data.norm(2).item() ** 2 for param in group['params'] if param.requires_grad])
            #     total_norm += total_norm_params
            # total_norm = total_norm ** .5
            # clip_multiplier = min(self.max_per_sample_grad_norm / (total_norm + 1e-8), 1.)

            # for group in self.param_groups:
            #     for param, accum_grad in zip(group['params'], group['aggregate_grads']):
            #         if param.requires_grad:
            #             accum_grad.add_(param.grad.data.mul(clip_multiplier))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['aggregate_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def add_noise_(self):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:

                        # Accumulate gradients
                        param.grad.data = accum_grad.clone()

                        # Add noise and update grads
                        # See: https://github.com/facebookresearch/pytorch-dp/blob/master/torchdp/privacy_engine.py
                        noise = _generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param)
                        param.grad += noise / self.batch_size

                        # See alternative below
                        # param.grad.data.add_(_generate_noise(self.noise_multiplier, self.max_per_sample_grad_norm, param) / self.batch_size)

        def step(self, *args, **kwargs):
            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer

# Adam optimizer
AdamDP = create_optimizer(Adam)

