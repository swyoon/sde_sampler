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
                 data_path=None,
                 val_data_path: Optional[str] = None,
                 test_data_path: Optional[str] = None):
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

        if val_data_path is not None:
            val_data = np.load(val_data_path, allow_pickle=True)
            self.val_data = remove_mean(torch.tensor(val_data, dtype=torch.float32),
                                    self.n_particles,
                                    self.n_dims)
            self.n_val_data = val_data.shape[0]
            print(f"Val Ground truth sample shape: {val_data.shape}")
        else:
            self.val_data = None
            self.n_val_data = 0
            print("No Val ground truth sample provided")

        if test_data_path is not None:
            test_data = np.load(test_data_path, allow_pickle=True)
            self.test_data = remove_mean(torch.tensor(test_data, dtype=torch.float32),
                                    self.n_particles,
                                    self.n_dims)
            self.n_test_data = test_data.shape[0]
            print(f"Test Ground truth sample shape: {test_data.shape}")
        else:
            self.test_data = None
            self.n_test_data = 0
            print("No Test ground truth sample provided")

    def energy(self, x):
        batch_shape = x.shape[0]
        x = x.view(batch_shape, self.n_particles, self.n_dims)

        dists = distances_from_vectors(x)
        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        lj_energies = lj_energies.view(batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1))
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        lj_energies = torch.clamp(lj_energies,min=-1e8,max=1e8)

        return lj_energies[:, None]

    def _remove_mean(self, x):
        return x - torch.mean(x, dim=1, keepdim=True)

    def unnorm_log_prob(self, x):
        return -self.energy(x)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                # TODO: should it be _gt_disc or _disc
                #energy_output = self.energy_function(copy_x).sum()
                self.energy(copy_x).sum().backward()
                lgv_data = -copy_x.grad.data
            return lgv_data

    def sample(self, shape: tuple, mode='train'):
        assert len(shape) == 1
        if(mode == 'train'):
            assert self.data is not None, "No ground truth data available"
            n_samples = shape[0]
            index = np.random.choice(self.n_data, n_samples, replace=False)
            return self.data[index]
        if(mode == 'val'):
            assert self.val_data is not None, "No ground truth data available"
            n_samples = shape[0]
            index = np.random.choice(self.n_val_data, n_samples, replace=False)
            return self.val_data[index]
        if(mode == 'test'):
            assert self.test_data is not None, "No ground truth data available"
            n_samples = shape[0]
            index = np.random.choice(self.n_test_data, n_samples, replace=False)
            return self.test_data[index]
        
    def to(self, device):
        super().to(device)
        if self.data is not None:
            self.data = self.data.to(device)
            self.val_data = self.val_data.to(device)
            self.test_data = self.test_data.to(device)
        return self