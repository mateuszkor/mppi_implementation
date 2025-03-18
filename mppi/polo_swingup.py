import equinox
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable
import time
import optax
import equinox as eqx
import wandb

from numpy.ma.core import inner
from nn.base_nn import Network

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

# wandb.init(project="polo_mppi", name="cartpole_experiment")

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
        running_cost_fn: fn(dx, u) -> cost (float)random_U

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

@equinox.filter_jit
def simulate_trajectory_mppi_gamma(mx, dx, set_control_fn, running_cost_fn, terminal_cost_fn, value_net, U, gamma = 0.99):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        dx: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    def step_fn(carry, u):
        dx, t = carry
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx) * (gamma ** t)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return (dx, t+1), (state, c)

    (dx_final, H), (states, costs) = jax.lax.scan(step_fn, (dx, 0), U)
    state_final = jnp.concatenate([dx_final.qpos, dx_final.qvel])
    total_cost = jnp.sum(costs) + value_net(state_final) * (gamma ** len(U))
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

class ValueNN(Network):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        return self.layers[-1](x).squeeze()

import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, timestep, state, control_sequence):
        """Store (state, control sequence) in buffer."""
        experience = (timestep, state, control_sequence)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """Randomly sample a batch from buffer."""
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []


@dataclass
class POLO:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]  # Gradient of the loss function with respect to controls
    lam: float  # Temperature parameter used for weighting the control rollouts
    U_init: jnp.ndarray
    running_cost: Callable[[jnp.ndarray], float]
    terminal_cost: Callable[[jnp.ndarray], float]
    set_control: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mx: mjx.Model

    #NN
    replay_buffer: ReplayBuffer
    value_net: ValueNN
    value_optimizer: optax.GradientTransformation
    value_opt_state: optax.OptState
    update_frequency: int
    mini_batch: int
    grad_steps: int

    def mppi_solver(self, dx, U, key, t):
        # print("\nMPPI Solver")
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, N_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi_gamma, in_axes=(None, None, None, None, None, None, 0))
        x_batch, cost_batch = simulate_trajectory_batch(mx, dx_internal, set_control, running_cost, terminal_cost, value_net, U_rollouts)

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)

        optimal_U = U + weighted_controls
        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        # print(f"Optimal Cost: {self.loss(optimal_U)}")
        states, total_cost = simulate_trajectory_mppi(mx, dx_internal, set_control, running_cost, terminal_cost, optimal_U)
        # print(f"Total Cost: {total_cost}")

        state = jnp.concatenate([dx_internal.qpos, dx_internal.qvel])
        replay_buffer.add(t, state, optimal_U)  # Store experience in buffer
        
        return optimal_U[0], next_U
    
    def mppi_target(self, state, U, key, U_opt=None):
        # print("\nMPPI Target")

        nq = mx.nq  
        qpos = state[:nq]
        qvel = state[nq:]

        dx = mjx.make_data(mx)
        dx_internal = dx.replace(qpos=dx.qpos.at[:].set(qpos), 
                    qvel=dx.qvel.at[:].set(qvel))

        split_keys = jax.random.split(key, N_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi_gamma, in_axes=(None, None, None, None, None, None, 0))
        x_batch, cost_batch = simulate_trajectory_batch(mx, dx_internal, set_control, running_cost, terminal_cost, value_net, U_rollouts)

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1
        # print(f"weights {weights.shape}")

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)
        optimal_U = U + weighted_controls
        # print(f"Optimal Cost: {self.loss(optimal_U)}")

        _, target_cost = simulate_trajectory_mppi_gamma(mx, dx_internal, set_control, running_cost, terminal_cost, value_net, optimal_U)
        # print(f"Target Cost: {target_cost}")
        
        return target_cost
        
    def update_value_function(self, states, target_values):
        def loss_fn(value_net):
            predicted_values = jax.vmap(value_net)(states)
            return jnp.mean((predicted_values - target_values)**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(self.value_net)
        updates, self.value_opt_state = self.value_optimizer.update(grads, self.value_opt_state)
        self.value_net = eqx.apply_updates(self.value_net, updates)
        return loss

if __name__ == "__main__":
    path = "xmls/cartpole.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    qpos_init = jnp.array([0.0, 3.14])
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))

    Nsteps, nu, N_rollouts = 100, mx.nu, 20
    update_frequency, mini_batch, grad_steps = 50,20,1

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return 1e-1 * (dx.qpos[0] ** 2 + angle_cost ** 2) + 1e-2 * jnp.sum(dx.ctrl ** 2)
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return 1 * (dx.qpos[0] ** 2 + angle_cost ** 2)

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    task_completed = False

    replay_buffer = ReplayBuffer(capacity=100000)
    key_nn = jax.random.PRNGKey(0)
    value_net = ValueNN(dims=[4,64,64,1], key=key_nn)
    value_optimizer = optax.adam(1e-3)
    value_opt_state = value_optimizer.init(eqx.filter(value_net, eqx.is_array))

    U = jnp.zeros((Nsteps, nu))
    # U = jnp.ones((Nsteps, nu)) * 1.0
    polo = POLO(loss=loss_fn, grad_loss=grad_loss_fn, lam=0.8, U_init=U, running_cost=running_cost, terminal_cost=terminal_cost, set_control=set_control, mx=mx,
                replay_buffer=replay_buffer, value_net=value_net, value_optimizer=value_optimizer, value_opt_state=value_opt_state, update_frequency=update_frequency, mini_batch=mini_batch, grad_steps=grad_steps)
    key = jax.random.PRNGKey(0)

    import mujoco
    import mujoco.viewer
    data_cpu = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data_cpu)
    import numpy as np

    i = 1
    
    @equinox.filter_jit
    def jit_step(mx, dx):
        return mjx.step(mx, dx)
    
    with viewer as v:
        while not task_completed:
            # Select action accorfing to MPPI
            key, subkey = jax.random.split(key)
            u0, U = polo.mppi_solver(dx, U, subkey, i)
            dx = set_control(dx, u0)
            dx = jit_step(mx, dx)
            # print(f"Step {i}: qpos={dx.qpos}, qvel={dx.qvel}")

            data_cpu.qpos[:] = np.array(jax.device_get(dx.qpos))
            data_cpu.qvel[:] = np.array(jax.device_get(dx.qvel))
            mujoco.mj_forward(model, data_cpu)
            v.sync()  
            
            if i % polo.update_frequency == 0:
                for _ in range(polo.grad_steps):
                    # Sample n states from replay buffer
                    batch = polo.replay_buffer.sample(polo.mini_batch)
                    if not batch:
                        continue  # Skip if buffer isn't full enough
                    
                    timesteps, states, control_sequences = zip(*batch)
                    timesteps = jnp.array(timesteps)
                    states = jnp.array(states)
                    control_sequences = jnp.array(control_sequences)

                    key = jax.random.PRNGKey(0)  # Initialize PRNG key
                    split_keys = jax.random.split(key, polo.mini_batch)
                    random_U = jax.random.normal(key, (Nsteps, nu))  # Shape: (Nsteps, nu)

                    generate_trajectory_targets = jax.vmap(polo.mppi_target, in_axes=(0, None, 0, None))
                    targets = generate_trajectory_targets(states, random_U, split_keys, control_sequences)
                    # print(f"targets: {targets}")
                    
                    value_loss = polo.update_value_function(states, targets)
                    print(f'Value function loss: {value_loss}')
                    # wandb.log({"Value Loss": float(value_loss), "Step": i})

            # if jnp.mod(dx.qpos[1], 2*jnp.pi) < 0.1:
            #     print(dx.qpos[0], dx.qpos[1])
            #     task_completed = True

            i += 1