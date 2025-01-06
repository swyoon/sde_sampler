python scripts/main.py solver=pis wandb.project=funnel_pis target=funnel

python scripts/main.py solver=dis model@generative_ctrl=clipped generative_ctrl.base_model.channels=32 +lr_scheduler=multi_step wandb.project=funnel_dis target=funnel

python scripts/main.py solver=dds model@generative_ctrl=clipped generative_ctrl.base_model.channels=32 +lr_scheduler=multi_step wandb.project=funnel_dds target=funnel

