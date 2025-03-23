# main.py
import equinox
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

from mujoco import mjx
import mujoco.viewer
import numpy as np
import contextlib

from config.config import load_config, generate_name
from models.mppi import MPPI
from models.polo import POLO
from models.simulation import make_loss
from simulations.simulation_constructor import SimulationConstructor
from algorithms.algorithm_constructor import OptimizerConstructor
import wandb

import optax
import equinox as eqx
from nn.base_nn import Network, ValueNN
from dataclasses import asdict
from utils.replay_buffer import ReplayBuffer

def run_simulation(config, headless=False, use_wandb=False, algorithm="vanilla_mppi"):

    # Create simulation components using factory

    simulation_config = {
        'simulation': {
            'path': config.simulation.path,
            'sensors': config.simulation.sensors,
            'algo': config.simulation.algo
        },
        'costs': {
            'control_weight': config.costs.control_weight,
            'quat_weight': config.costs.quat_weight,
            'finger_weight': config.costs.finger_weight,
            'terminal_weight': config.costs.terminal_weight,
            'intermediate_weight': config.costs.intermediate_weight
        }
    }

    if config.hand:
        simulation_config['hand'] = {
            'qpos_init': config.hand.qpos_init,
            'goal_quat': config.hand.goal_quat  # Keep it as jnp.ndarray
        }

    # Load MuJoCo model
    path = config.simulation.path
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)

    # Initialize random key
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    qpos_init, is_completed, get_log_data, set_control, running_cost, terminal_cost = SimulationConstructor.create_simulation(
        config.simulation.name, simulation_config, mx, subkey
    )

    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    print(f'Inittial qpos: {dx.qpos}')
    
    # Setup MPPI controller
    Nsteps, nu = config.mppi.n_steps, mx.nu
    N_rollouts = config.mppi.n_rollouts
    
    # Create loss function
    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    # Initialize control sequence
    if config.mppi.initial_control == "zeros":
        U_init = jnp.zeros((Nsteps, nu))
    elif config.mppi.initial_control == "ones":
        U_init = jnp.ones((Nsteps, nu)) 
    elif config.mppi.initial_control == "random":
        key, subkey = jax.random.split(key)
        U_init = jax.random.normal(subkey, (Nsteps, nu))
    else:
        raise ValueError(f"Unknown initial control: {config.mppi.initial_control}")
    
    # Create optimizer
    optimizer = MPPI(
        loss=loss_fn, 
        grad_loss=grad_loss_fn, 
        lam=config.mppi.lambda_value, 
        running_cost=running_cost, 
        terminal_cost=terminal_cost, 
        set_control=set_control, 
        mx=mx,
        n_rollouts=N_rollouts,
        sim=config.simulation.name,
        baseline=config.mppi.baseline,
        sim_traj_mppi_func=None,
    )

    # optimizer, next_U = OptimizerConstructor.create_optimizer(
    #     algorithm,
    #     config,
    #     mx,
    #     loss_fn,
    #     grad_loss_fn,
    #     running_cost,
    #     terminal_cost,
    #     set_control,
    #     key
    # )


    key, key_nn = jax.random.split(key)
    replay_buffer = ReplayBuffer(capacity=100000)
    value_net = ValueNN(dims=[4,64,64,1], key=key_nn)
    value_optimizer = optax.adam(1e-3)
    value_opt_state = value_optimizer.init(eqx.filter(value_net, eqx.is_array))
    update_frequency, mini_batch, grad_steps = 20,20,5
    net_update_type = "random"  #or optimal (or possibly just value)

    optimizer = POLO(
        loss=loss_fn, 
        grad_loss=grad_loss_fn, 
        lam=config.mppi.lambda_value, 
        running_cost=running_cost, 
        terminal_cost=terminal_cost, 
        set_control=set_control, 
        mx=mx,
        n_rollouts=N_rollouts,
        sim=config.simulation.name,
        baseline=config.mppi.baseline,
        sim_traj_mppi_func=None,
        replay_buffer=replay_buffer,
        value_net=value_net,
        value_optimizer=value_optimizer,
        value_opt_state=value_opt_state,
        update_frequency=update_frequency,
        mini_batch=mini_batch,
        grad_steps=grad_steps,
        gamma=1.0,
        net_update_type=net_update_type
    )

    # Setup viewer
    if not headless:
        data = mujoco.MjData(model)
        data.qpos[:] = np.array(jax.device_get(dx.qpos))
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        viewer = contextlib.nullcontext() 
    
    # JIT-compile step function
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)
    
    # Main simulation loop
    with viewer as v:
        i = 1
        task_completed = False
        next_U = U_init
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)
            
            # Get control and update control sequence
            u0, next_U, optimal_cost, separate_costs = optimizer.solver(dx, next_U, subkey, i)
            
            dx = set_control(dx, u0)
            # Step simulation
            dx = jit_step(mx, dx)

            jax.debug.print("optimal_cost = {x}", x=optimal_cost)
            # log for wandb here
            if use_wandb:
                print("Logging to wandb")
                log_data = get_log_data(separate_costs, optimal_cost, i, dx.qpos)
                wandb.log(log_data)
                # wandb.log({"test": 13, "Step": i})

            #polo
            if algorithm == "polo" and i % optimizer.update_frequency == 0:
                for _ in range(optimizer.grad_steps):
                    batch = optimizer.replay_buffer.sample(optimizer.mini_batch)
                    if not batch:
                        continue  # Skip if buffer isn't full enough
                    
                    timesteps, states, control_sequences = zip(*batch)
                    timesteps = jnp.array(timesteps)
                    states = jnp.array(states)
                    opt_control_sequences = jnp.array(control_sequences)

                    if optimizer.net_update_type == "random":
                        key, subkey = jax.random.split(key)
                        update_U = jax.random.normal(subkey, (optimizer.mini_batch, Nsteps, nu))  # Shape: (Nsteps, nu)
                    else:
                        update_U = opt_control_sequences

                    key, subkey = jax.random.split(key)
                    split_keys = jax.random.split(subkey, optimizer.mini_batch)
                    
                    generate_trajectory_targets = jax.vmap(optimizer.mppi_target, in_axes=(0, 0, 0))
                    targets = generate_trajectory_targets(states, update_U, split_keys)
                    
                    print(f"targets: {targets}")
                    value_loss = optimizer.update_value_function(states, targets)
                    print(f'Value function loss: {value_loss}')
                    # wandb.log({"Value Loss": float(value_loss), "Step": i})

            # Update viewer
            if not headless:
                data.qpos[:] = np.array(jax.device_get(dx.qpos))
                data.qvel[:] = np.array(jax.device_get(dx.qvel))
                mujoco.mj_forward(model, data)
                v.sync()
            
            # if is_completed(dx.qpos, 1.0, True):
            #     print("Task completed seccessfully")
            #     print(f'Final qpos: {dx.qpos}')
            #     task_completed = True

            if i == 2000: 
                print("Task reached iteration limit")
                task_completed = True
            i += 1

if __name__ == "__main__":
    algorithm = "polo"   #vanilla_mppi, polo
    simulation = "swingup"    #swingup, hand_fixed or hand_free

    config, config_dict = load_config(f"config/{algorithm}/{simulation}.yaml")
    config.print_config()

    headless = 0
    use_wandb = 0
    if use_wandb:
        name = generate_name(config_dict)
        wandb.init(config=config, project="mppi_vanilla_hand_fixed_nosensors", name=name, mode="offline")

    run_simulation(config, headless, use_wandb, algorithm=algorithm)
    try: 
        pass        

    except KeyboardInterrupt:
        print("Exiting from the simulation")
    
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if use_wandb:
            print("Finishing wandb run")
            wandb.finish()

