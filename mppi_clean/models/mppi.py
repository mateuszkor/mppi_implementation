import equinox
import jax
import jax.numpy as jnp
from mujoco import mjx
from dataclasses import dataclass
from typing import Callable
from models.simulation import simulate_trajectory_mppi

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
    n_rollouts: int
    use_baseline: bool = False

    def solver(self, dx, U, key):
        dx_internal = jax.tree.map(lambda x: x, dx)

        split_keys = jax.random.split(key, self.n_rollouts)
        noise = jax.vmap(lambda subkey: jax.random.normal(subkey, (U.shape[0], self.mx.nu)))(split_keys)
        U_rollouts = jnp.expand_dims(U, axis=0) + noise

        simulate_trajectory_batch = jax.vmap(simulate_trajectory_mppi, in_axes=(None, None, None, None, None, 0))
        x_batch, cost_batch = simulate_trajectory_batch(
            self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, U_rollouts
        )

        if self.use_baseline:
            baseline = jnp.min(cost_batch)
            cost_batch -= baseline

        weights = jnp.exp(-cost_batch / self.lam) 
        weights /= jnp.sum(weights)  # Normalize the weights to sum to 1

        weighted_controls = jnp.einsum('k,kij->ij', weights, noise)

        optimal_U = U + weighted_controls
        _, optimal_cost = simulate_trajectory_mppi(self.mx, dx_internal, self.set_control, self.running_cost, self.terminal_cost, optimal_U)

        next_U = jnp.roll(optimal_U, shift=-1, axis=0) 
        return optimal_U[0], next_U, optimal_cost
