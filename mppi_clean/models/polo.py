import equinox as eqx
import jax
import jax.numpy as jnp
from mujoco import mjx
from dataclasses import dataclass
from typing import Callable
from models.simulation import simulate_trajectory_mppi_gamma, simulate_trajectory_mppi_hand
from utils.replay_buffer import ReplayBuffer
import optax
from nn.base_nn import Network, ValueNN
from numpy.ma.core import inner
from models.mppi import MPPI


@dataclass
class POLO(MPPI):
    #NN
    replay_buffer: ReplayBuffer
    value_net: ValueNN
    value_optimizer: optax.GradientTransformation
    value_opt_state: optax.OptState
    update_frequency: int
    mini_batch: int
    grad_steps: int
    gamma: float
    net_update_type: str

    def __post_init__(self):
        # This method is automatically called after __init__
        if self.sim == "cartpole":
            self.sim_traj_mppi_func = simulate_trajectory_mppi_gamma
        else:
            self.sim_traj_mppi_func = simulate_trajectory_mppi_hand

    def solver(self, dx, U, key, t):
        # print("\nMPPI Solver")
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, self.n_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], self.mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(self.sim_traj_mppi_func, in_axes=(None, None, None, None, None, None, 0, None))
        _, cost_batch, _ = simulate_trajectory_batch(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, self.value_net, U_rollouts, self.gamma
        )

        # crucial for hand to use baseline, otherwise the cost will be NaN
        if self.baseline:
            baseline = jnp.min(cost_batch)
            cost_batch -= baseline

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)

        optimal_U = U + weighted_controls
        
        _, optimal_cost, separate_costs = self.sim_traj_mppi_func(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, self.value_net, optimal_U, self.gamma, True
        )

        # add to the replay buffer
        state = jnp.concatenate([dx_internal.qpos, dx_internal.qvel])
        self.replay_buffer.add(t, state, optimal_U)  # Store experience in buffer
        
        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        return optimal_U[0], next_U, optimal_cost, separate_costs
    
    def mppi_target(self, state, U, key):
        # print("\nMPPI Target")
        nq = self.mx.nq  
        qpos = state[:nq]
        qvel = state[nq:]

        dx = mjx.make_data(self.mx)
        dx_internal = dx.replace(qpos=dx.qpos.at[:].set(qpos), qvel=dx.qvel.at[:].set(qvel))

        split_keys = jax.random.split(key, self.n_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], self.mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise
 
        simulate_trajectory_batch = jax.vmap(self.sim_traj_mppi_func, in_axes=(None, None, None, None, None, None, 0, None))
        _, cost_batch, _ = simulate_trajectory_batch(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, self.value_net, U_rollouts, self.gamma
        )

        if self.baseline:
            baseline = jnp.min(cost_batch)
            cost_batch -= baseline

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)
        optimal_U = U + weighted_controls

        _, target_cost, _ = self.sim_traj_mppi_func(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, self.value_net, optimal_U, self.gamma,
        )
        
        return target_cost
    
    def update_value_function(self, states, target_values):
        def loss_fn(value_net, states, target_values):
            predicted_values = jax.vmap(lambda x: value_net(x))(states)
            return jnp.mean((predicted_values - target_values)**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=False)(self.value_net, states, target_values)

        updates, new_opt_state = self.value_optimizer.update(grads, self.value_opt_state)
        self.value_net = eqx.apply_updates(self.value_net, updates)
        self.value_opt_state = new_opt_state
        
        return loss