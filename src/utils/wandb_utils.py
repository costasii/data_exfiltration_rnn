import os

# set defaults if not already in environment
os.environ.setdefault("WANDB_PROJECT", "lsb-encoding")
os.environ.setdefault("WANDB_ENTITY", "anon")
os.environ.setdefault("WANDB_MODE", "online")

def wb_init_optional(cfg, **kwargs):
    if os.getenv("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
        return wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            mode=os.getenv("WANDB_MODE"),
            config=cfg,
            **kwargs
        )
    except Exception:
        return None
    
def flatten_dict(d, parent_key="", sep="/"):
    items = {}
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items