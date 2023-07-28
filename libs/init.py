import torch
import argparse
import libs.utils as utils
import wandb


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", default=False, help="sync with W&B")
    args = parser.parse_args()
    WANDB = args.wandb
    CONFIG = utils.load_yaml()
    if WANDB:
        wprj = wandb.init(
            project=CONFIG.wandb.project,
            name=CONFIG.wandb.name,
            resume=False,
            config=CONFIG,
        )
        RUN_ID = wprj.id
    else:
        RUN_ID = utils.get_random_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return RUN_ID, CONFIG, WANDB, device
