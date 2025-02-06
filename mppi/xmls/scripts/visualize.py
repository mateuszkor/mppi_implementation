import time
import os
import subprocess
import pathlib
import importlib
import yaml
import argparse

import argcomplete
from paramiko import SSHClient
import jax.numpy as jnp
import mujoco
from mujoco import viewer

import jax
from brax.io import model
from brax.envs.fd import get_environment
from brax.training.acme import running_statistics

remote_config_filename = "resolved_config.yaml"
script_dir = os.path.dirname(os.path.realpath(__file__))

config_dir = os.path.join(script_dir, "remote_cfg")
param_dir = os.path.join(script_dir, "params")

def get_most_recent_run_parameters():
    usename = "alexaldermanwebb"
    address = "129.215.91.20"
    
    client = SSHClient()
    client.load_system_host_keys()
    client.connect(address, username=usename)
    stdin, stdout, stderr = client.exec_command('./brax/brax/scripts/newest_train_parameters.sh')
    parameter_path = pathlib.Path(stdout.read().strip().decode("utf-8"))
    train_dir = parameter_path.parents[1]

    isotime = f"{train_dir.parents[0].name}T{train_dir.name}"

    output_config = os.path.join(config_dir, isotime)
    os.makedirs(output_config, exist_ok=True)
    subprocess.call(["sh", "./sync_config.sh", train_dir / "cfg" / remote_config_filename, output_config])

    local_params = os.path.join(param_dir, isotime)
    os.makedirs(local_params, exist_ok=True)
    output = subprocess.check_output([
        f"script -q -c 'scp {usename}@{address}:{parameter_path} {local_params}' </dev/null",
    ], shell=True)

    policy_params_name = output.decode('utf-8').strip().split(" ")[0]

    return \
        os.path.join(output_config, remote_config_filename), \
        os.path.join(local_params, policy_params_name)

def visualise_traj_generic(
    x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01
):
    with viewer.launch_passive(m, d) as v:
        for dx in x:
            step_start = time.time()
            d.qpos[:] = dx.qpos
            d.qvel[:] = dx.qvel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def generate_trajctory(env_reset, env_step, inference_fn):
    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = env_reset(rng=rng)
    for _ in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = inference_fn(state.obs, act_rng)
        state = env_step(state, act)

    return rollout

def get_paths():
    parser = argparse.ArgumentParser(description='FD visualization script.')
    argcomplete.autocomplete(parser)
    parser.add_argument('-p','--param-path', help='Path to policy parameters.')
    parser.add_argument('-c','--config-path', help='Path to training configuration.')
    args = parser.parse_args()

    if args.config_path and args.param_path:
        return args.config_path, args.param_path

    return get_most_recent_run_parameters()

def main():
    run_config_path, policy_parameters_path = get_paths()
    
    with open(run_config_path, "r") as f:
        run_config = yaml.safe_load(f)

    policy_params = model.load_params(policy_parameters_path)
    policy_params = (
        policy_params[0],
        policy_params[1].policy
    )

    alg_module = importlib.import_module(run_config["alg"]["make_inference_path"])
    
    make_network = getattr(alg_module, run_config["alg"]["make_network_name"])
    make_inference = getattr(alg_module, "make_inference_fn")

    env = get_environment(run_config["env"]["name"])
    jit_env_reset = jax.jit(env.reset)

    rng = jax.random.PRNGKey(seed=1)
    env_state = jit_env_reset(rng)
    obs_shape = env_state.obs.shape

    make_inference_fn = make_inference(
        make_network(obs_shape, env.action_size, preprocess_observations_fn=running_statistics.normalize)
    )
    inference_fn = make_inference_fn(policy_params)

    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    trajectory = generate_trajctory(jit_env_reset, jit_env_step, jit_inference_fn)

    data = mujoco.MjData(env.model)
    visualise_traj_generic(trajectory, data, env.model)

if __name__=='__main__':
    main()
