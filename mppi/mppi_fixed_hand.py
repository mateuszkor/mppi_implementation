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
import wandb

from numpy.ma.core import inner

wandb.init(project="mppi_hand", name="hand_free_", mode="offline")

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U, weights):
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
        c = running_cost_fn(dx, weights[0], weights[1])
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))

    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final, weights[2])

    # jax.debug.print("costs: {x}", x=jnp.sum(costs))
    # jax.debug.print("terminal: {x}", x=terminal_cost_fn(dx_final, weights[2]))

    return states, total_cost

@equinox.filter_jit
def simulate_trajectory_mppi(mx, dx, set_control_fn, running_cost_fn, terminal_cost_fn, U, weights, final=False):
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
        c = running_cost_fn(dx, weights[0], weights[1], optimal=final)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx_final, (states, costs) = jax.lax.scan(step_fn, dx, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final, weights[2])
    if final:
        jax.debug.print("costs: {x}", x=jnp.sum(costs))
        jax.debug.print("terminal: {x}", x=terminal_cost_fn(dx_final, weights[2]))
    return None, total_cost

def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U, weights):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U, weights
        )
        return total_cost
    return loss

def quaterion_diff(q1,q2):
    q1_norm = q1 / jnp.linalg.norm(q1)
    q1_conj = jnp.array([q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3]])
    s1, x1, y1, z1 = q1_conj
    s2, x2, y2, z2 = q2
    return jnp.array([
        s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2, 
        s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2,  
        s1 * y2 - x1 * z2 + y1 * s2 + z1 * x2,  
        s1 * z2 + x1 * y2 - y1 * x2 + z1 * s2   
    ])

@dataclass
class MPPI:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]  # Gradient of the loss function with respect to controls
    lam: float  # Temperature parameter used for weighting the control rollouts
    running_cost: Callable[[jnp.ndarray], float]
    terminal_cost: Callable[[jnp.ndarray], float]
    set_control: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mx: mjx.Model
    ws: jnp.ndarray

    def solver(self, dx, U, key):
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, N_rollouts)
        noise = jax.vmap(lambda subkey: 1. * jax.random.normal(subkey, (U.shape[0], mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise
        
        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi, in_axes=(None, None, None, None, None, 0, None))
        _, cost_batch = simulate_trajectory_batch(mx, dx_internal, set_control, running_cost, terminal_cost, U_rollouts, self.ws)

        baseline = jnp.min(cost_batch)
        weights = jnp.exp(-(cost_batch - baseline) / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)
        
        optimal_U = U + weighted_controls

        _, optimal_cost = simulate_trajectory_mppi(mx, dx_internal, set_control, running_cost, terminal_cost, optimal_U, self.ws, final=True)
        print(f"Optimal Cost: {optimal_cost}")
        # print(f"Optimal Cost: {self.loss(optimal_U, self.ws)}")

        wandb.log({"Cost": float(optimal_cost), "Step": i})

        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        
        return optimal_U[0], next_U

if __name__ == "__main__":
    path = "xmls/shadow_hand/scene_right.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3) 
    nq = mx.nq
    qpos_init = jax.random.uniform(subkey1, mx.nq, minval=-0.1, maxval=0.1)
    # qvel_init = jax.random.uniform(subkey2, mx.nv, minval=-0.05, maxval=0.05)

    thumb_touch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "thumb_touch")
    if thumb_touch_id == -1:
        raise RuntimeError("Thumb touch sensor not found in model")
    print(thumb_touch_id)

    qpos_init = jnp.array([
        -0.21292259, -0.15179611, -0.15403237,  0.49449011,  0.68263494,  0.55026304,
        -0.08796999,  0.25227709,  0.88539273,  0.69736412, -0.02815096,  0.82066547,
        0.59070077,  0.91741982,  0.27842981,  0.03551317,  0.81599353,  0.65583756,
        0.71857939, -0.15087653,  0.67113446,  0.03214713,  0.00824811,  0.8125244,
        1.0,  0.0, 0.0, 0.0,  1., 0., 0., 0.
    ])
    
    qpos_init = jnp.array([ -0.17355183, -0.56166065,  -0.035,  0.82,  0.89, 0.6, 
        -0.02693524,  0.19608329,  0.6657893,   1.1829618,  -0.00787788,  0.5805705,
        0.9559763,   0.8349045,   0.1953071,   0.08200635,  0.44465777,  1.1225411,  
        0.8899891,   -0.05, 1.2,  0.09,  0.7,   0.2530998, 
        1.,          0.,          0.,          0.,          
        1.,          0.,          0.,          0.
    ])

    qpos_init = jnp.array([
        -0.03,   -0.36,   -0.35,    0.82,    0.89,    0.61,   
        -0.021,   1.1,    0.41,    0.88,   -0.074,   0.67,    
        1.4,   -0.0024,   0.43,   -0.34,    0.54,    1.1,     
        0.39,   0,   1.2,    0.089,    0.7,     0.6,    
        0.92,    0.2,    -0.32,    0.03,    
        1.,     0.,      0.,      0.
    ])
    
    goal_quat = model.qpos0[(nq-4):nq]
    curr_quat = model.qpos0[(nq-8):(nq-4)]


    qpos_init = qpos_init.at[(nq-8):nq].set(model.qpos0[(nq-8):nq])
    # qvel_init = qvel_init.at[24:30].set(model.qvel0[24:30])
    # dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    # dx = dx.replace(qvel=dx.qvel.at[:].set(qvel_init))
    
    Nsteps, nu, N_rollouts = 100, mx.nu, 100
    goal_quat = jnp.array([0.0,0.0,1.0,0.0])
    weights = jnp.array([1e-4, 0.5, 1.0])

    print(f'dx.qpos: {dx.qpos}')
    print(f'Ball init quat: {curr_quat}')
    print(f'Ball goal quat: {goal_quat}')

    def set_control(dx, u):
        forces = u + dx.qpos[:(nq-11)]
        return dx.replace(ctrl=dx.ctrl.at[:].set(forces))

    def running_cost(dx, ctrl_weight, quat_weight, optimal=False):
        u = dx.ctrl
        ctrl_cost = ctrl_weight * jnp.sum(u ** 2)

        ball_quat = dx.qpos[(nq-8):(nq-4)]
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        quat_cost = quat_weight * (angle ** 2)
        # # jax.debug.print("quat_cost: {x}", x=quat_cost)

        thumb_sensor_data = dx.sensordata[0]
        thumb_contact_cost = 1/(thumb_sensor_data + (1/100))
        # jax.debug.print("thumb_cost: {x}", x = thumb_contact_cost)

        finger_data, palm_data = dx.sensordata[0:5], dx.sensordata[5]
        finger_cost = jnp.sum(1/(finger_data + (1/100))) * 0.01
        palm_cost = jnp.sum(1/(palm_data + (1/100)))

        # if optimal:
        #     jax.debug.print("finger_cost: {x}", x=finger_cost)
        #     jax.debug.print("palm_cost: {x}", x=finger_cost)
        #     jax.debug.print("ctrl_cost: {x}", x=ctrl_cost)
        #     jax.debug.print("quat_cost: {x}", x=quat_cost)
            
        # return finger_cost
        return ctrl_cost + quat_cost + finger_cost

    def terminal_cost(dx, quat_weight):
        # -------- ball pos -------------
        # ball_position = dx.qpos[24:27]
        # jax.debug.print("cur pos: {x}", x=ball_position)
        # goal_position =jnp.array([0.45, 0.0, 0.15]) 
    
        # pos_cost = 10. * jnp.sum((ball_position - goal_position) ** 2)
        # jax.debug.print("pos cost: {y}", y=pos_cost)

        # -------- ball quats -----------
        ball_quat = dx.qpos[24:28]
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        quat_cost = quat_weight * (angle ** 2)

        # jax.debug.print("quat_cost: {x}", x=quat_cost)
        return quat_cost

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    task_completed = False

    U = jax.random.normal(key, (Nsteps, nu))  # Shape: (Nsteps, nu)
    U = jnp.zeros((Nsteps, nu))
    optimizer = MPPI(loss=loss_fn, grad_loss=grad_loss_fn, lam=0.5, running_cost=running_cost, terminal_cost=terminal_cost, set_control=set_control, mx=mx, ws=weights)

    import mujoco
    import mujoco.viewer

    data_cpu = mujoco.MjData(model)
    headless = 0
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data_cpu)
    else:
        viewer = contextlib.nullcontext() 
    
    import numpy as np
    i = 1
    
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)

    with viewer as v:
        while not task_completed:
            print(f"iteration: {i}")
            key, subkey = jax.random.split(key)

            u0, U = optimizer.solver(dx, U, subkey)

            dx = set_control(dx, u0)            
            dx = jit_step(mx, dx) #overflow here

            # print(f"Step {i}: qpos={dx.qpos}")
            ball_quat = dx.qpos[(nq-8):(nq-4)]
            print(f"ball_quat {i}: quat={ball_quat}")
            
            if not headless: 
                data_cpu.qpos[:] = np.array(jax.device_get(dx.qpos))
                data_cpu.qvel[:] = np.array(jax.device_get(dx.qvel))
                mujoco.mj_forward(model, data_cpu)
                v.sync()  
            
            i += 1

            
            val = jnp.sum((quaterion_diff(ball_quat, goal_quat) ** 2)[1:])
            print(val)
            if val < 0.001:
                print(f"finished")
                task_completed = True

wandb.finish()