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
from models.simulation import make_loss
from simulations.simulation_constructor import SimulationConstructor
import wandb

from dataclasses import asdict

def run_simulation(config, headless=False, use_wandb=False, batch_size=10, display_index=0):

    # Create simulation components using factory

    simulation_config = {
        'simulation': {
            'path': config.simulation.path,
            'sensors': config.simulation.sensors
        },
        'costs': {
            'control_weight': config.costs.control_weight,
            'quat_weight': config.costs.quat_weight,
            'finger_weight': config.costs.finger_weight,
            'terminal_weight': config.costs.terminal_weight
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

    qpos_init, is_completed, get_log_data, set_control, running_cost, terminal_cost = SimulationConstructor.create_simulation(
        config.simulation.name, simulation_config, mx
    )

    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    batch_dx = jax.tree.map(lambda x: jnp.stack([x] * batch_size), dx)
    
    # Setup MPPI controller
    Nsteps, nu = config.mppi.n_steps, mx.nu
    N_rollouts = config.mppi.n_rollouts
    
    # Create loss function
    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))
    
    # Initialize random key
    key = jax.random.PRNGKey(0)

    # Initialize control sequence
    if config.mppi.initial_control == "zeros":
        U_batch = jnp.zeros((batch_size, Nsteps, nu))
    elif config.mppi.initial_control == "ones":
        U_batch = jnp.ones((batch_size, Nsteps, nu)) 
    elif config.mppi.initial_control == "random":
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        U_batch = jax.vmap(lambda k: jax.random.normal(k, (Nsteps, nu)))(keys)
    else:
        raise ValueError(f"Unknown initial control: {config.mppi.initial_control}")
    
    # Create MPPI optimizer
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
        baseline=config.mppi.baseline
    )
    
    # Setup viewer
    if not headless:
        data = mujoco.MjData(model)
        data.qpos[:] = np.array(jax.device_get(batch_dx.qpos[display_index]))
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        viewer = contextlib.nullcontext() 
    
    # JIT-compile step function
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)

    @jax.vmap
    def batch_set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
    # Main simulation loop
    i = 1
    task_completed = False
    next_U_batch = U_batch
    completed_list = jnp.array([False] * batch_size)
    first_index, first_completion_index = -1, -1

    with viewer as v:
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)
            split_keys = jax.random.split(subkey, batch_size)
            
            # Get control and update control sequence
            solver_batch = jax.vmap(optimizer.solver, in_axes=(0, 0, 0))
            u0_batch, next_U_batch, optimal_cost_batch, separate_costs_batch = solver_batch(batch_dx, next_U_batch, split_keys)
            
            batch_dx = batch_set_control(batch_dx, u0_batch)
            batch_dx = jax.vmap(jit_step, in_axes=(None, 0))(mx, batch_dx)
            # print("optimal_cost", optimal_cost_batch)

            # log for wandb here
            if use_wandb:
                # print("Logging to wandb")
                logger_batch = jax.vmap(get_log_data, in_axes=(0,0,None,0))
                log_data_batch = logger_batch(separate_costs_batch, optimal_cost_batch, i, batch_dx.qpos)
                wandb_dict = {}
                for k,val in log_data_batch.items():
                    k_dict = k
                    for index in range(batch_size):
                        val_dict = jax.device_get(val[index]).item()
                        wandb_dict[f"{k_dict}_{index}"] = val_dict
                wandb.log(wandb_dict)


            # Update viewer
            if not headless:
                data.qpos[:] = np.array(jax.device_get(batch_dx.qpos[display_index]))
                data.qvel[:] = np.array(jax.device_get(batch_dx.qvel[display_index]))
                mujoco.mj_forward(model, data)
                v.sync()
            
            is_completed_batch = jax.vmap(is_completed, in_axes=(0,None,None))
            any_completed = is_completed_batch(batch_dx.qpos, 1, False)
            
            if jnp.any(any_completed):
                for j in range(batch_size):
                    if any_completed[j] == True:
                        if first_index == -1: 
                            first_index = j
                            first_completion_index = i
                        completed_list = completed_list.at[j].set(True)

            if jnp.all(completed_list):
                print("All tasks completed successfully")
                print(f'first_index: {first_index} at iteration {first_completion_index}')
                task_completed = True

            if i == 2000: 
                print("Task reached iteration limit")
                task_completed = True
            i += 1

if __name__ == "__main__":
    algorithm = "vanilla_mppi"
    simulation = "swingup"    #swingup, hand_fixed or hand_free

    config, config_dict = load_config(f"config/{algorithm}/{simulation}.yaml")
    config.print_config()

    headless = False
    use_wandb = True
    batch_size, display_index = 50, 24
    if use_wandb:
        name = generate_name(config_dict) + "_batch_50"
        wandb.init(config=config, project="mppi_vanilla_batch", name=name, mode="offline")

    run_simulation(config, headless, use_wandb, batch_size, display_index)
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

