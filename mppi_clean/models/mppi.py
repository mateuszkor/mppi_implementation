import equinox
import jax
import jax.numpy as jnp
from mujoco import mjx
from dataclasses import dataclass
from typing import Callable
from models.simulation import simulate_trajectory_mppi, simulate_trajectory_mppi_hand

@dataclass
class MPPI:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]  # Gradient of the loss function with respect to controls
    lam: float  # Temperature parameter used for weighting the control rollouts
    running_cost: Callable[[jnp.ndarray], float]
    terminal_cost: Callable[[jnp.ndarray], float]
    set_control: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    mx: mjx.Model
    n_rollouts: int
    sim: str
    baseline: bool
    sim_traj_mppi_func: Callable  # Make this optional with a default value
    
    def __post_init__(self):
        # This method is automatically called after __init__
        if self.sim == "cartpole":
            self.sim_traj_mppi_func = simulate_trajectory_mppi
        else:
            self.sim_traj_mppi_func = simulate_trajectory_mppi_hand

    def solver(self, dx, U, key, t):
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, self.n_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], self.mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(self.sim_traj_mppi_func, in_axes=(None, None, None, None, None, 0))
        _, cost_batch, _ = simulate_trajectory_batch(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, U_rollouts
        )

        if self.baseline:
            baseline = jnp.min(cost_batch)
            cost_batch -= baseline

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)

        optimal_U = U + weighted_controls
        _, optimal_cost, separate_costs = self.sim_traj_mppi_func(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, optimal_U, final=True
        )

        next_U = U = jnp.roll(optimal_U, shift=-1, axis=0) 
        return optimal_U[0], next_U, optimal_cost, separate_costs
