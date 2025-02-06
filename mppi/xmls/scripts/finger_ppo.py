import os
import importlib
import functools
import hydra
import pathlib
from hydra.core.hydra_config import HydraConfig
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import List

import wandb
from tensorboardX import SummaryWriter

from jax import config
from brax.io import model
from brax.envs.fd import get_environment

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def create_wandb_run(
        project: str,
        group: str,
        entity: str,
        alg_name: str,
        env_name: str,
        seed: int,
        notes: str | None,
        job_config: dict, 
        run_id: str,
        resume: bool = False
    ):

    name = f"{alg_name}_{env_name}_sweep_{seed}"

    return wandb.init(
        project=project,
        config=job_config,
        group=group,
        entity=entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=resume,
    )

def get_time_from_path(path: str):
    parts = pathlib.Path(path).parts

    date = parts[-2]
    time = parts[-1]

    return f"{date}T{time}"

def progress(times: List[datetime], writer: SummaryWriter, num_steps: int, metrics: dict):
    times.append(datetime.now())

    log_scalar = lambda key, value: writer.add_scalar(f"{key}", value, num_steps)
    for reward in ["reward"]:
      log_scalar(f"eval/episode_{reward}", metrics[f"eval/episode_{reward}"].item())
      log_scalar(f"eval/episode_{reward}_std", metrics[f"eval/episode_{reward}_std"].item())
    log_scalar(f"eval/avg_episode_length", metrics[f"eval/avg_episode_length"].item())
    log_scalar(f"eval/epoch_eval_time", metrics[f"eval/epoch_eval_time"])
    log_scalar(f"eval/sps", metrics[f"eval/sps"])

    writer.flush()

def policy_params(save_directory: os.path, current_step: int, make_policy, params):
    save_path = os.path.join(save_directory, f"policy_params_step_{current_step}")
    model.save_params(save_path, params)

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    hydra_logdir = HydraConfig.get()["runtime"]["output_dir"]

    param_copy_dir = os.path.join(hydra_logdir, "resolved_config.yaml")
    OmegaConf.save(cfg, param_copy_dir)

    cfg_full = OmegaConf.to_container(cfg, resolve=True)
    create_wandb_run(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        entity=cfg.wandb.entity,
        alg_name=cfg.alg.name,
        env_name=cfg.env.name,
        seed=cfg.general.seed,
        notes=cfg.wandb.notes,
        job_config=cfg_full,
        run_id=get_time_from_path(hydra_logdir),
        resume=cfg.wandb.resume
    )

    logdir = os.path.join(hydra_logdir, cfg.general.logdir)
    writer = SummaryWriter(logdir)

    times = [datetime.now()]

    alg_module = importlib.import_module(cfg.alg.module_path)
    algorithm_train = getattr(alg_module, "train")

    param_dir = os.path.join(hydra_logdir, cfg.general.param_dir)
    os.makedirs(param_dir)

    env = get_environment(cfg.env.name)
    make_inference_fn, params, _ = algorithm_train(
        **cfg.alg.params,
        environment=env,
        progress_fn=functools.partial(progress, times, writer),
        policy_params_fn=functools.partial(policy_params, param_dir),
        seed=cfg.general.seed
    )

    wandb.finish()

if __name__=='__main__':
    train()
