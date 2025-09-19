# lennard_jones.py
from __future__ import annotations

import io
import logging
import math
from typing import Optional, Callable

import numpy as np
import torch

# plotting & image conversion for visualization helpers
from PIL import Image
import matplotlib.pyplot as plt

# import base Distribution and rejection_sampling from your repo
from .base import Distribution, rejection_sampling

# -------------------------
# utility functions
# -------------------------


def tile(a: torch.Tensor, dim: int, n_tile: int) -> torch.Tensor:
    """Tile `a` along dimension `dim` n_tile times (pure torch)."""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)

    order_index = torch.arange(init_dim, device=a.device)
    order_index = order_index.repeat_interleave(n_tile) + (
        torch.arange(n_tile, device=a.device).repeat(init_dim) * init_dim
    )
    return torch.index_select(a, dim, order_index)


def distance_vectors(x: torch.Tensor, remove_diagonal: bool = True) -> torch.Tensor:
    """
    Compute pairwise difference vectors r_{ij} = x_i - x_j.

    Returns:
        If remove_diagonal True: shape [batch, n_particles, n_particles - 1, n_dims]
        Else: [batch, n_particles, n_particles, n_dims]
    """
    # x: [batch, n_particles, n_dims]
    xi = x.unsqueeze(2)  # [batch, n_particles, 1, n_dims]
    xj = x.unsqueeze(1)  # [batch, 1, n_particles, n_dims]
    r = xi - xj  # [batch, n_particles, n_particles, n_dims]

    if remove_diagonal:
        n = x.shape[1]
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        # flatten first two dims for mask indexing, then reshape
        r = r[:, mask].view(-1, x.shape[1], x.shape[1] - 1, x.shape[2])
    return r


def distance_vectors_v2(x: torch.Tensor, y: torch.Tensor, remove_diagonal: bool = True) -> torch.Tensor:
    """
    Alternate implementation using tile that mirrors original repo's behavior.
    """
    r1 = tile(x.unsqueeze(2), 2, x.shape[1])
    r2 = tile(y.unsqueeze(2), 2, y.shape[1])
    r = r1 - r2.permute([0, 2, 1, 3])
    if remove_diagonal:
        n = x.shape[1]
        r = r[:, torch.eye(n, n, dtype=torch.bool, device=x.device) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distances_from_vectors(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert distance vectors to scalar distances. Input r shape [..., n_dims] -> output [...]."""
    return (r.pow(2).sum(dim=-1) + eps).sqrt()


def compute_distances(x: torch.Tensor, n_particles: int, n_dimensions: int, remove_duplicates: bool = True) -> torch.Tensor:
    """
    Compute pairwise distances (uses torch.cdist).
    If remove_duplicates True returns flattened upper-triangle distances for each batch sample
    shape [batch, n_particles * (n_particles - 1) // 2]
    Else returns full [batch, n_particles, n_particles]
    """
    x = x.reshape(-1, n_particles, n_dimensions)
    distances = torch.cdist(x, x)  # [batch, n, n]
    if remove_duplicates:
        n = n_particles
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=distances.device), diagonal=1)
        distances = distances[:, mask].reshape(-1, n * (n - 1) // 2)
    return distances


def remove_mean(samples: torch.Tensor, n_particles: int, n_dimensions: int) -> torch.Tensor:
    """Make configuration mean-free (zero center of mass)."""
    shape = samples.shape
    samples = samples.view(-1, n_particles, n_dimensions)
    samples = samples - torch.mean(samples, dim=1, keepdim=True)
    samples = samples.view(*shape)
    return samples


# -------------------------
# Lennard-Jones energy
# -------------------------


def lennard_jones_energy_torch(r: torch.Tensor, eps: float = 1.0, rm: float = 1.0) -> torch.Tensor:
    """
    Lennard-Jones pairwise energy as function of scalar distance r.
    r can be tensor.
    """
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(Distribution):
    """
    A Distribution-like wrapper that exposes the Lennard-Jones energy as an unnormalized log-probability.

    This inherits your project's `Distribution` base and implements `unnorm_log_prob`.
    """
    def __init__(self, dim, n_particles,n_dims, eps=1.0, rm=1.0,
                 oscillator=True,
                 oscillator_scale=1.0, energy_factor=1.0,
                 data_path=None):
        super().__init__(dim=dim)
        self.n_particles = n_particles
        self.n_dims = n_dims
        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale
        self._energy_factor = energy_factor
        #self.stddevs = torch.tensor([0.6807141304016113])

        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data),
                                    self.n_particles,
                                    self.n_dims)
            self.n_data = data.shape[0]
            print(f"Ground truth sample shape: {data.shape}")
        else:
            self.data = None
            self.n_data = 0
            print("No Ground truth sample provided")

    def _energy(self, x):
        batch_shape = x.shape[0]
        x = x.view(batch_shape, self.n_particles, self.n_dims)

        dists = distances_from_vectors(x)
        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        lj_energies = lj_energies.view(batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1))
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        return x - torch.mean(x, dim=1, keepdim=True)

    def unnorm_log_prob(self, x):
        return -self._energy(x)

    # def score(self, x: torch.Tensor,eps=1.0, sigma=1.0, *args, **kwargs) -> torch.Tensor:
    #     """
    #     Compute score function (∇ log p(x)) for Lennard-Jones N-particle system.

    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Shape [batch, N*d] or [batch, N, d].
    #     eps : float
    #         Depth of potential well (ε).
    #     sigma : float
    #         Characteristic length scale (σ).

    #     Returns
    #     -------
    #     score : torch.Tensor
    #         Same shape as x, detached (no gradient history).
    #     """
    #     if x.dim() == 2:  # [batch, N*d] → reshape
    #         batch, nd = x.shape
    #         N = nd // 3
    #         x = x.view(batch, N, 3)

    #     # pairwise differences
    #     rij = x[:, :, None, :] - x[:, None, :, :]   # [batch, N, N, 3]
    #     dist = torch.norm(rij, dim=-1)   
        
    #     print(dist)           # [batch, N, N]

    #     mask = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)

    #     # avoid /0
    #     dist = torch.where(mask, dist, torch.ones_like(dist))
        

    #     inv_r2 = (sigma / dist) ** 2
    #     inv_r6 = inv_r2 ** 3
    #     inv_r12 = inv_r6 ** 2

    #     # scalar prefactor for force magnitude
    #     coef = 24 * eps * (2 * inv_r12 - inv_r6) / dist**2   # [batch, N, N]

    #     # force on i due to j
    #     fij = coef[:, :, :, None] * rij   # [batch, N, N, 3]

    #     # sum over j≠i
    #     grad_E = fij.sum(dim=2)           # [batch, N, 3]

    #     # score = -∇E
    #     score = -grad_E

    #     return score.view(x.shape[0], -1).detach()

    def sample(self, shape: tuple):
        assert len(shape) == 1
        assert self.data is not None
        n_samples = shape[0]
        index = np.random.choice(self.n_data, n_samples, replace=False)
        return self.data[index]

    def to(self, device):
        super().to(device)
        if self.data is not None:
            self.data = self.data.to(device)
        return self
    


# ####
# """IDEM Score Estimator"""

# import numpy as np
# import torch


# ## Change this to our energy function
# from dem.energies.base_energy_function import BaseEnergyFunction

# ###Import from dem
# from dem.models.components.clipper import Clipper
# from dem.models.components.noise_schedules import BaseNoiseSchedule


# def wrap_for_richardsons(score_estimator):
#     def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
#         bigger_samples = score_estimator(t, x, energy_function, noise_schedule, num_mc_samples)

#         smaller_samples = score_estimator(
#             t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
#         )

#         return (2 * bigger_samples) - smaller_samples

#     return _fxn


# def log_expectation_reward(
#     t: torch.Tensor,
#     x: torch.Tensor,
#     energy_function: BaseEnergyFunction,
#     noise_schedule: BaseNoiseSchedule,
#     num_mc_samples: int,
#     clipper: Clipper = None,
# ):
#     repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
#     repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

#     h_t = noise_schedule.h(repeated_t).unsqueeze(1)

#     samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

#     log_rewards = energy_function(samples)

#     if clipper is not None and clipper.should_clip_log_rewards:
#         log_rewards = clipper.clip_log_rewards(log_rewards)

#     return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)


# def estimate_grad_Rt(
#     t: torch.Tensor,
#     x: torch.Tensor,
#     energy_function: BaseEnergyFunction,
#     noise_schedule: BaseNoiseSchedule,
#     num_mc_samples: int,
# ):
#     if t.ndim == 0:
#         t = t.unsqueeze(0).repeat(len(x))

#     grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
#     vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None), randomness="different")

#     return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)



# ##DEM Clipper
# from typing import Optional

# import torch

# _EPSILON = 1e-6

# class Clipper:
#     def __init__(
#         self,
#         should_clip_scores: bool,
#         should_clip_log_rewards: bool,
#         max_score_norm: Optional[float] = None,
#         min_log_reward: Optional[float] = None,
#     ):
#         self._should_clip_scores = should_clip_scores
#         self._should_clip_log_rewards = should_clip_log_rewards
#         self.max_score_norm = max_score_norm
#         self.min_log_reward = min_log_reward

#     @property
#     def should_clip_scores(self) -> bool:
#         return self._should_clip_scores

#     @property
#     def should_clip_log_rewards(self) -> bool:
#         return self._should_clip_log_rewards

#     def clip_scores(self, scores: torch.Tensor) -> torch.Tensor:
#         score_norms = torch.linalg.vector_norm(scores, dim=-1).detach()

#         clip_coefficient = torch.clamp(self.max_score_norm / (score_norms + _EPSILON), max=1)

#         return scores * clip_coefficient.unsqueeze(-1)

#     def clip_log_rewards(self, log_rewards: torch.Tensor) -> torch.Tensor:
#         return log_rewards.clamp(min=self.min_log_reward)

#     def wrap_grad_fxn(self, grad_fxn):
#         def _run(*args, **kwargs):
#             scores = grad_fxn(*args, **kwargs)
#             if self.should_clip_scores:
#                 scores = self.clip_scores(scores)

#             return scores

#         return _run


# ##DEM Noise Scheduler
# class BaseNoiseSchedule(ABC):
#     @abstractmethod
#     def g(t):
#         # Returns g(t)
#         pass

#     @abstractmethod
#     def h(t):
#         # Returns \int_0^t g(t)^2 dt
#         pass