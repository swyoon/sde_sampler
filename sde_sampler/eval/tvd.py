import torch

def interatomic_dist(samples,n_particles,n_dim):
    samples = samples.reshape(-1,n_particles,n_dim)
    batch_size, n_particles, _ = samples.shape
    n_particles = samples.shape[-2]
    print(samples.shape)
    # Compute the pairwise differences and distances
    distances = samples[:, None, :, :] - samples[:, :, None, :]
    distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]

    dist = torch.linalg.norm(distances, dim=-1)

    return dist

def Energy_TVD(gen_sample, gt_sample, energy):
    gt_energy = energy(gt_sample).detach().cpu()
    gen_energy = energy(gen_sample).detach().cpu()
    H_data_set, x_dataset = torch.histogram(gt_energy, bins=200)
    H_gen_samples, _ = torch.histogram(gen_energy, bins=(x_dataset))
    tv = 0.5*(torch.abs(H_data_set/H_data_set.sum() - H_gen_samples/H_gen_samples.sum())).sum()
    return tv

def Atomic_TVD(gen_sample, gt_sample,distr):
    if not (hasattr(distr, "n_particles") and hasattr(distr, "n_dims")):
        return None
    
    n_particles = distr.n_particles
    n_dims = distr.n_dims

    gt_interatomic = interatomic_dist(gt_sample,n_particles,n_dims).detach().cpu()
    gen_interatomic = interatomic_dist(gen_sample,n_particles,n_dims).detach().cpu()
    H_data_set, x_dataset = torch.histogram(gt_interatomic, bins=200)
    H_gen_samples, _ = torch.histogram(gen_interatomic, bins=(x_dataset))
    tv = 0.5*(torch.abs(H_data_set/H_data_set.sum() - H_gen_samples/H_gen_samples.sum())).sum()
    return tv