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
        a: float = 0.0,
        b: float = -4.0,
        c: float = 0.9,
        tau: float = 1.0,
        d0: float = 1.0,
        data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
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

        # load ground-truth samples if provided
        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data, dtype=torch.float32),
                                    self.n_particles,
                                    self.n_dims)
            self.n_data = data.shape[0]
            print(f"Train Ground truth sample shape: {data.shape}")
        else:
            self.data = None
            self.n_data = 0
            print("No Train ground truth sample provided")

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

    def pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between particles."""
        batch_size = x.shape[0]
        #print(f"x requires grad : {x.requires_grad}")
        coords = x.view(batch_size, self.n_particles, self.n_dims)  # (B,N,d)
        #print(f"coords requires grad : {coords.requires_grad}")
        coords_front = coords.unsqueeze(2)
        coords_back = coords.unsqueeze(1)
        #print(f"coords front requires grad : {coords_front.requires_grad}")
        #print(f"coords back requires grad : {coords_back.requires_grad}")
        diff = coords_front - coords_back # (B,N,N,d)
        #print(f"diff requires grad : {diff.requires_grad}")
        dij = torch.norm(diff, dim=-1)  # (B,N,N)
        idx_i, idx_j = torch.triu_indices(self.n_particles,
                                          self.n_particles, offset=1)
        #print(f"pairwise requires grad : {dij.requires_grad}")
        return dij[:, idx_i, idx_j]  # (B, n_pairs)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total energy of configuration batch."""
        dij = self.pairwise_distances(x)
        diff = dij - self.d0
        energy = (
            self.a * diff +
            self.b * diff**2 +
            self.c * diff**4
        ).sum(dim=-1) / (2 * self.tau)

        #energy = torch.clamp(energy,min=-1000,max=1000)
        return energy

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.energy(x).unsqueeze(-1)

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
    
    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)
    
    def to(self, device):
        super().to(device)
        if self.data is not None:
            self.data = self.data.to(device)
            self.val_data = self.val_data.to(device)
            self.test_data = self.test_data.to(device)
        return self