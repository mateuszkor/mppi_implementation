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

def run_simulation(config, headless=False, use_wandb=False):

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

    qpos_init, set_control, running_cost, terminal_cost = SimulationConstructor.create_simulation(
        config.simulation.name, simulation_config, mx
    )

    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    print(dx.qpos)
    
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
        U_init = jnp.zeros((Nsteps, nu))
    elif config.mppi.initial_control == "random":
        U_init = jax.random.normal(key, (Nsteps, nu))
    else:
        raise ValueError(f"Unknown initial control: {config.mppi.initial_control}")
    
    # Create MPPI optimizer
    optimizer = MPPI(
        loss=loss_fn, 
        grad_loss=grad_loss_fn, 
        lam=config.mppi.lambda_value, 
        U_init=U_init, 
        running_cost=running_cost, 
        terminal_cost=terminal_cost, 
        set_control=set_control, 
        mx=mx,
        n_rollouts=N_rollouts,
        baseline=config.mppi.baseline
    )
    
    # Setup viewer
    if not headless:
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        viewer = contextlib.nullcontext() 
    
    # JIT-compile step function
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)
    
    # Main simulation loop
    i = 1
    task_completed = False
    next_U = U_init
    
    with viewer as v:
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)
            
            # Get control and update control sequence
            u0, next_U, optimal_cost = optimizer.solver(dx, next_U, subkey)
            dx = set_control(dx, u0)
            print("optimal_cost", optimal_cost)
            
            # Step simulation
            dx = jit_step(mx, dx)
            if config.simulation.name == "swingup":
                print(f"Step {i}: qpos={dx.qpos}, qvel={dx.qvel}")
            else:
                ball_quat = dx.qpos[(mx.nq-8):(mx.nq-4)]
                print(f"ball_quat {i}: quat={ball_quat}")

            # log for wandb here
            if use_wandb:
                print("Logging to wandb")
                log_data = {"optimal_cost": float(optimal_cost), "Step": i}
                wandb.log(log_data)

            # Update viewer
            if not headless:
                data.qpos[:] = np.array(jax.device_get(dx.qpos))
                data.qvel[:] = np.array(jax.device_get(dx.qvel))
                mujoco.mj_forward(model, data)
                v.sync()
            
            i += 1
            
            # Uncomment to check for task completion
            # if jnp.mod(dx.qpos[1], 2*jnp.pi) < 0.1:
            #     print(dx.qpos[0], dx.qpos[1])
            #     task_completed = True

if __name__ == "__main__":
    algorithm = "vanilla_mppi"
    simulation = "hand_fixed"

    config, config_dict = load_config(f"config/{algorithm}/{simulation}.yaml")
    config.print_config()

    headless = False
    use_wandb = False
    if use_wandb:
        name = generate_name(config=config)
        wandb.init(config=config, project="mppi_vanilla", name=name, mode="offline")


    try: 
        run_simulation(config, headless, use_wandb)

    except KeyboardInterrupt:
        print("Exiting from the simulation")
    
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if use_wandb:
            print("Finishing wandb run")
            wandb.finish()

