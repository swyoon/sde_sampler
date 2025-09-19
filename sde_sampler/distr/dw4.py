import torch
import numpy as np
from typing import Optional
from .base import Distribution

def remove_mean(samples: torch.Tensor, n_particles: int, n_dimensions: int) -> torch.Tensor:
    """Make configuration mean-free (zero center of mass)."""
    shape = samples.shape
    samples = samples.view(-1, n_particles, n_dimensions)
    samples = samples - torch.mean(samples, dim=1, keepdim=True)
    samples = samples.view(*shape)
    return samples


class DW4(Distribution):
    """
    4-particle double-well potential system.

    Energy:
        E(x) = Σ_{i<j} [ a*(dij-d0) + b*(dij-d0)^2 + c*(dij-d0)^4 ] / (2τ)

    Parameters
    ----------
    n_particles : int
        Number of particles (default 4).
    particle_dim : int
        Dimensionality of each particle (default 2, so total dim=8).
    """
    def __init__(
        self,
        n_particles: int = 4,
        n_dims: int = 2,
        dim: int = 8,
        domain_delta: float = 3.0,
        a: float = 0.0,
        b: float = -4.0,
        c: float = 0.9,
        tau: float = 1.0,
        d0: float = 1.0,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.n_particles = n_particles
        self.n_dims = n_dims

        self.a = a
        self.b = b
        self.c = c
        self.tau = tau
        self.d0 = d0

        # domain box for each coordinate
        if self.domain is None:
            domain = domain_delta * torch.tensor([[-1.0, 1.0]])
            domain = domain.repeat(dim, 1)   # (dim, 2)
            self.set_domain(domain)

        # load ground-truth samples if provided
        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data, dtype=torch.float32),
                                    self.n_particles,
                                    self.n_dims)
            self.n_data = data.shape[0]
            print(f"Ground truth sample shape: {data.shape}")
        else:
            self.data = None
            self.n_data = 0
            print("No ground truth sample provided")

    def _pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between particles."""
        batch_size = x.shape[0]
        coords = x.view(batch_size, self.n_particles, self.n_dims)  # (B,N,d)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,d)
        dij = torch.norm(diff, dim=-1)  # (B,N,N)
        idx_i, idx_j = torch.triu_indices(self.n_particles,
                                          self.n_particles, offset=1)
        return dij[:, idx_i, idx_j]  # (B, n_pairs)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total energy of configuration batch."""
        dij = self._pairwise_distances(x)
        diff = dij - self.d0
        energy = (
            self.a * diff +
            self.b * diff**2 +
            self.c * diff**4
        ).sum(dim=-1) / (2 * self.tau)
        return energy

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self._energy(x).unsqueeze(-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # same, since normalizing constant unknown
        return self.unnorm_log_prob(x)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """∇ log p(x) via autograd."""
        x = x.requires_grad_(True)
        logp = self.unnorm_log_prob(x).sum()
        grad, = torch.autograd.grad(logp, x)
        return grad

    def sample(self, shape: tuple):
        assert len(shape) == 1
        assert self.data is not None, "No ground truth data available"
        n_samples = shape[0]
        index = np.random.choice(self.n_data, n_samples, replace=False)
        return self.data[index]

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)
    
