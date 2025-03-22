# models/simulation.py
import equinox
import jax
import jax.numpy as jnp
from mujoco import mjx
from typing import Callable

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
def simulate_trajectory_mppi(mx, dx, set_control_fn, running_cost_fn, terminal_cost_fn, U, final=False):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        dx: MuJoCo data
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

    dx_final, (_, costs) = jax.lax.scan(step_fn, dx, U)
    terminal_cost = terminal_cost_fn(dx_final)
    total_cost = jnp.sum(costs) + terminal_cost
    separate_costs = None
    if final:
        separate_costs = (jnp.sum(costs), terminal_cost)
        # jax.debug.print("costs: {x}", x=jnp.sum(costs))
        # jax.debug.print("terminal: {x}", x=terminal_cost)

    separate_costs = (jnp.sum(costs), terminal_cost_fn(dx_final))
    return None, total_cost, separate_costs

@equinox.filter_jit
def simulate_trajectory_mppi_hand(mx, dx, set_control_fn, running_cost_fn, terminal_cost_fn, U, final=False):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        dx: MuJoCo data
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
        ctrl_c, quat_c, finger_c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, ctrl_c, quat_c, finger_c)

    dx_final, (_, ctrl_costs, quat_costs, finger_costs) = jax.lax.scan(step_fn, dx, U)
    running_cost, terminal_cost = jnp.sum(ctrl_costs + quat_costs + finger_costs), terminal_cost_fn(dx_final)
    total_cost = running_cost + terminal_cost
    
    separate_costs = None
    if final:
        separate_costs = (jnp.sum(ctrl_costs), jnp.sum(quat_costs), jnp.sum(finger_costs), running_cost, terminal_cost)
        jax.debug.print("ctrl_costs: {x}", x=jnp.sum(ctrl_costs))
        jax.debug.print("quat_costs: {x}", x=jnp.sum(quat_costs))
        jax.debug.print("finger_costs: {x}", x=jnp.sum(finger_costs))
        jax.debug.print("running_cost: {x}", x=running_cost)
        jax.debug.print("terminal_cost: {x}", x=terminal_cost)

    return None, total_cost, separate_costs

def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U
        )
        return total_cost
    return loss


