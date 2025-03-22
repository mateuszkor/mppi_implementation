import equinox
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable
import time
import contextlib

from numpy.ma.core import inner

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        qpos_init: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    def step_fn(dx, u):
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost

@equinox.filter_jit
def simulate_trajectory_mppi(mx, dx, set_control_fn, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        qpos_init: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    def step_fn(dx, u):
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx_final, (states, costs) = jax.lax.scan(step_fn, dx, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost

def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U):
        _, total_cost = simulate_trajectory( #should this be pmp?
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U
        )
        return total_cost
    return loss

@dataclass
class MPPI:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]  # Gradient of the loss function with respect to controls
    lam: float  # Temperature parameter used for weighting the control rollouts
    running_cost: Callable[[jnp.ndarray], float]
    terminal_cost: Callable[[jnp.ndarray], float]
    set_control: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mx: mjx.Model

    def solver(self, dx, U, key):
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, N_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise
        # jax.debug.print("U_rollouts for dx={}: {}", dx.qpos, U_rollouts)

        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi, in_axes=(None, None, None, None, None, 0))
        x_batch, cost_batch = simulate_trajectory_batch(mx, dx_internal, set_control, running_cost, terminal_cost, U_rollouts)

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)

        optimal_U = U + weighted_controls
        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        print(f"Optimal Cost: {self.loss(optimal_U)}")

        return optimal_U[0], next_U

if __name__ == "__main__":
    path = "xmls/cartpole.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    Nsteps, nu, N_rollouts, batch_size = 100, mx.nu, 10, 10

    dx = mjx.make_data(mx)
    qpos_init = jnp.array([0.0, 3.14])
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))

    batch_dx = jax.tree.map(lambda x: jnp.stack([x] * batch_size), dx)

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return 1 * (dx.qpos[0] ** 2 + angle_cost ** 2)

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))
    task_completed = False

    U_batch = jnp.ones((batch_size, Nsteps, nu)) 
    optimizer = MPPI(loss=loss_fn, grad_loss=grad_loss_fn, lam=0.8, running_cost=running_cost, terminal_cost=terminal_cost, set_control=set_control, mx=mx)
    key = jax.random.PRNGKey(0)

    import mujoco
    import mujoco.viewer
    data_cpu = mujoco.MjData(model)
    headless = True
    viewer = contextlib.nullcontext() if headless else mujoco.viewer.launch_passive(model, data_cpu)
    import numpy as np

    i = 1
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)
    
    @jax.vmap
    def batch_set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
    with viewer as v:
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)
            split_keys = jax.random.split(subkey, batch_size)

            solver_batch = jax.vmap(optimizer.solver, in_axes=(0, 0, 0))
            u0_batch, U_batch = solver_batch(batch_dx, U_batch, split_keys)

            batch_dx = batch_set_control(batch_dx, u0_batch)
            batch_dx = jax.vmap(jit_step, in_axes=(None, 0))(mx, batch_dx)

            print(f"Step {i}: qpos={batch_dx.qpos[0]}, qvel={batch_dx.qvel[0]}")

            data_cpu.qpos[:] = np.array(jax.device_get(batch_dx.qpos[0]))
            data_cpu.qvel[:] = np.array(jax.device_get(batch_dx.qvel[0]))
            mujoco.mj_forward(model, data_cpu)
            if not headless: v.sync() 
            i += 1

            for j in range(batch_size):
                if jnp.mod(batch_dx.qpos[j][1], 2*jnp.pi) < 0.1:
                    print(f"Finished in the batch number: {j}")
                    print(batch_dx.qpos[j][0], batch_dx.qpos[j][1])
                    task_completed = True
        
        