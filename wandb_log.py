import wandb
api = wandb.Api()
ENTITY, PROJECT = "KAIST_RISE", "Gamma_Sweep_Wrapper_v3"   # 예: "jjeonghyeon755-org", "myo_alin5"

for run in api.runs(f"{ENTITY}/{PROJECT}"):
    for f in run.files():
        if f.name == "output.log":   # 또는 f.name.endswith(".log")
            try:
                f.delete()
                print("deleted:", run.id, f.name)
            except Exception as e:
                print("skip:", run.id, f.name, e)
