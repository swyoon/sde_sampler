"""
Script for running the experiments.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path


import hydra
import wandb
import yaml
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from sde_sampler.utils import hydra as hydra_utils
from sde_sampler.utils.wandb import merge_wandb_cfg

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig):
    logging.info("---------------------------------------------------------------")
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("out_dir"):
        cfg.out_dir = str(out_dir)
    logging.info("Solver output path: %s", cfg.out_dir)

    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    # Setup pkeops to prevent CUDA errors
    # if cfg.get("keops_build_path") is not None:
    #     import pykeops

    #     keops_path = Path(cfg.keops_build_path).absolute()
    #     keops_path.mkdir(parents=True, exist_ok=True)
    #     pykeops.set_build_folder(str(keops_path))
    #     logging.info("Pykeops build path: %s", keops_path)

    # run solver

    # === Remove wandb-related keys ===
    if "wandb" in cfg:
        logging.info("Removing top-level wandb section from cfg")
        del cfg["wandb"]

    # Sometimes wandb appears in nested configs (logging, monitor, etc.)
    for k in list(cfg.keys()):
        if "wandb" in k.lower():
            logging.info(f"Removing wandb key: {k}")
            del cfg[k]

    # Log config and overrides (after cleaning)
    logging.info("---------------------------------------------------------------")
    logging.info("Run config (wandb keys removed):\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    solver = instantiate(cfg.solver, cfg)
    solver.setup()

    #checkpoint_path = "./sde_sampler_ckpt/ckpt022500.pt"
    #solver.load_checkpoint(str(checkpoint_path))
    n_trials = 5
    batch_size = cfg.eval_batch_size
    print(f"sampling Batch Size : {batch_size}")
    import time

    device = "cuda"
    prior_samples = solver.prior.sample((batch_size,)).to(device)
    ts = solver.eval_timesteps(device=prior_samples.device)

    time_list = []
    logging.info("Starting evaluation...")

    for i in range(n_trials):
        start = time.perf_counter()
        with torch.no_grad():
            samples, _, traj = solver.loss.simulate(
                ts=ts,
                x=prior_samples,
                terminal_unnorm_log_prob=solver.clipped_target_unnorm_log_prob,
                reference_log_prob=solver.reference_distr.log_prob,
                compute_ito_int=True,
                return_traj=True,
            )
        elapsed = time.perf_counter() - start
        logging.info(f"[Trial {i+1}] Sampling time: {elapsed:.4f} sec")
        time_list.append(elapsed)

    avg_time = np.mean(time_list)
    logging.info(f"Average sampling time over {n_trials} runs: {avg_time:.4f} sec")

if __name__ == "__main__":
    main()
