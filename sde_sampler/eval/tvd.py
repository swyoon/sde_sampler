import torch

def interatomic_dist(samples):
    #if samples.dim()==2:
    #    samples = samples.unsqueeze(0)
    #batch_size, n_particles, _ = samples.shape
    n_particles = samples.shape[-2]
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

def Atomic_TVD(gen_sample, gt_sample):
    gt_interatomic = interatomic_dist(gt_sample).detach().cpu()
    gen_interatomic = interatomic_dist(gen_sample).detach().cpu()
    H_data_set, x_dataset = torch.histogram(gt_interatomic, bins=200)
    H_gen_samples, _ = torch.histogram(gen_interatomic, bins=(x_dataset))
    tv = 0.5*(torch.abs(H_data_set/H_data_set.sum() - H_gen_samples/H_gen_samples.sum())).sum()
    return tv