import torch
import ot as pot
import math


def wasserstein(x0, x1): # Used the same code as in the DEM repository "dem/models/components/optimal_transport.py"
    M = torch.cdist(x0, x1)
    M = M**2
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    return math.sqrt(ret)