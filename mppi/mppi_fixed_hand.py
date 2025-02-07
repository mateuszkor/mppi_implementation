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
    dx0 = jax.tree.map(upscale, dx0)
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)

    jax.debug.print("costs: {x}", x=jnp.sum(costs))
    jax.debug.print("terminal: {x}", x=terminal_cost_fn(dx_final))

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

    dx = jax.tree.map(upscale, dx)
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost

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

@dataclass
class MPPI:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]  # Gradient of the loss function with respect to controls
    lam: float  # Temperature parameter used for weighting the control rollouts
    U_init: jnp.ndarray
    running_cost: Callable[[jnp.ndarray], float]
    terminal_cost: Callable[[jnp.ndarray], float]
    set_control: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mx: mjx.Model

    def solver(self, dx, U, key):
        # dx_internal = dx.replace(qpos=jnp.copy(dx.qpos), qvel=jnp.copy(dx.qvel))
        dx_internal = jax.tree.map(lambda x: x, dx)
        # dx_internal = dx.replace(qpos=dx.qpos.copy(), qvel=dx.qvel.copy())
        
        # key, subkey = jax.random.split(key)
        # noise = jax.random.normal(subkey, (U.shape[0], mx.nu))
        # U_rollouts = U + noise

        split_keys = jax.random.split(key, N_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi, in_axes=(None, None, None, None, None, 0))
        x_batch, cost_batch = simulate_trajectory_batch(mx, dx_internal, set_control, running_cost, terminal_cost, U_rollouts)

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1
        # print(f"weights {weights.shape}")

        # weighted_controls = jnp.tensordot(weights, U_rollouts, axes=([0], [0]))
        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)
        # print(f"weigthted_controls: {weighted_controls.shape}")

        optimal_U = U + weighted_controls
        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        print(f"Optimal Cost: {self.loss(optimal_U)}")
        
        return optimal_U[0], next_U
        
        # return solver

if __name__ == "__main__":
    path = "xmls/shadow_hand/scene_right.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    key = jax.random.PRNGKey(0)

    qpos_init = jax.random.uniform(key, 32, minval=-0.1, maxval=0.1)
    qpos_init = jnp.array([
        -0.21292259, -0.15179611, -0.15403237,  0.49449011,  0.68263494,  0.55026304,
        -0.08796999,  0.25227709,  0.88539273,  0.69736412, -0.02815096,  0.82066547,
        0.59070077,  0.91741982,  0.27842981,  0.03551317,  0.81599353,  0.65583756,
        0.71857939, -0.15087653,  0.67113446,  0.03214713,  0.00824811,  0.8125244,
        0.53514186,  0.84334721, -0.04544574,  0.0179821 ,  1., 0., 0., 0.
    ])

    qpos_init = qpos_init.at[24:32].set(model.qpos0[24:32])

    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    

    Nsteps, nu, N_rollouts = 100, mx.nu, 50

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        u = dx.ctrl
        ctrl_cost = 0
        # jax.debug.print("dx control: {x}", x=u)
        ctrl_cost = 1e-4 * jnp.sum(u ** 2)

        ball_quat = dx.qpos[24:28]
        goal_quat = jnp.array([0.0,1.0,0.0,0.0])
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        quat_cost = 0.1 * (angle ** 2)

        # jax.debug.print("ctrl_cost: {x}", x=ctrl_cost)
        # jax.debug.print("quat_cost: {x}", x=quat_cost)
        return ctrl_cost + quat_cost
    
    def quaterion_diff(q1,q2):
        q1_norm = q1 / jnp.linalg.norm(q1)
        q1_conj = jnp.array([q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3]])
        s1, x1, y1, z1 = q1_conj
        s2, x2, y2, z2 = q2
        return jnp.array([
            s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
            s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2,  # x
            s1 * y2 - x1 * z2 + y1 * s2 + z1 * x2,  # y
            s1 * z2 + x1 * y2 - y1 * x2 + z1 * s2   # z
        ])

    def terminal_cost(dx):
        # -------- ball pos -------------
        ball_position = dx.qpos[24:27]
        # jax.debug.print("cur pos: {x}", x=ball_position)
        # goal_position =jnp.array([0.45, 0.0, 0.15]) 
    
        # pos_cost = 10. * jnp.sum((ball_position - goal_position) ** 2)
        # jax.debug.print("pos cost: {y}", y=pos_cost)

        # -------- ball quats -----------
        ball_quat = dx.qpos[24:28]
        # jax.debug.print("cur ball quat: {x}", x=ball_quat)
        goal_quat = jnp.array([0.0,1.0,0.0,0.0])
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        # jax.debug.print("quat_diff: {x}", x=quat_diff)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        # jax.debug.print("angle: {x}", x=angle)
        quat_cost = 10 * (angle ** 2)

        # jax.debug.print("quat_cost: {x}", x=quat_cost)
        return quat_cost

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    task_completed = False

    U = jnp.zeros((Nsteps, nu))
    # U = jnp.ones((Nsteps, nu)) * 1.0
    optimizer = MPPI(loss=loss_fn, grad_loss=grad_loss_fn, lam=1, U_init=U, running_cost=running_cost, terminal_cost=terminal_cost, set_control=set_control, mx=mx)

    import mujoco
    import mujoco.viewer
    data_cpu = mujoco.MjData(model)
    # viewer = mujoco.viewer.launch_passive(model, data_cpu)
    headless = True
    viewer = contextlib.nullcontext() if headless else mujoco.viewer.launch_passive(model, data_cpu)
    import numpy as np

    i = 1
    
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)
    
    with viewer as v:
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)
            # make_mppi_solver = optimizer.make_MPPI_solver(mx)

            u0, U = optimizer.solver(dx, U, subkey)
            dx = set_control(dx, u0)
            dx = jit_step(mx, dx)

            # ball_pos = dx.qpos[24:28]
            ball_pos = dx.qpos
            # ball_quat = dx.qpos[27:31]    
            print(f"ball_quat {i}: quat={ball_pos}")
            # print(f"ball_quat {i}: quat={ball_quat}")

            data_cpu.qpos[:] = np.array(jax.device_get(dx.qpos))
            data_cpu.qvel[:] = np.array(jax.device_get(dx.qvel))
            mujoco.mj_forward(model, data_cpu)
            if not headless: v.sync()  
            i += 1
            
            # if jnp.mod(dx.qpos[1], 2*jnp.pi) < 0.1:
            #     print(dx.qpos[0], dx.qpos[1])
            #     task_completed = True