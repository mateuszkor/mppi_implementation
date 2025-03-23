import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple, Dict, Any
from models.mppi import MPPI
from models.polo import POLO
from utils.replay_buffer import ReplayBuffer
from nn.base_nn import ValueNN

class OptimizerConstructor:
    @staticmethod
    def create_optimizer(
        algorithm: str,
        config: Dict[str, Any],
        mx,
        loss_fn,
        grad_loss_fn,
        running_cost,
        terminal_cost,
        set_control,
        key
    ) -> Tuple[Any, jnp.ndarray]:
        """
        Factory method to create an optimizer based on the specified algorithm.

        Args:
            algorithm (str): The algorithm to use ("vanilla_mppi" or "polo").
            config (Dict[str, Any]): Configuration dictionary.
            mx: MuJoCo model representation.
            loss_fn: Loss function for optimization.
            grad_loss_fn: Gradient of the loss function.
            running_cost: Running cost function.
            terminal_cost: Terminal cost function.
            set_control: Function to apply control to the simulation.
            key: JAX random key.

        Returns:
            Tuple[Any, jnp.ndarray]: The optimizer and the initial control sequence.
        """
        
        N_rollouts = config['mppi']['n_rollouts']
        Nsteps, nu = config['mppi']['n_steps'], mx.nu

        if algorithm == "vanilla_mppi":
            optimizer = MPPI(
                loss=loss_fn,
                grad_loss=grad_loss_fn,
                lam=config['mppi']['lambda_value'],
                running_cost=running_cost,
                terminal_cost=terminal_cost,
                set_control=set_control,
                mx=mx,
                n_rollouts=N_rollouts,
                sim=config['simulation']['name'],
                baseline=config['mppi']['baseline'],
                sim_traj_mppi_func=None,
            )
        elif algorithm == "polo":
            key, key_nn = jax.random.split(key)
            
            # Replay buffer
            replay_buffer = ReplayBuffer(capacity=100000)
            
            # Value function network
            value_net = ValueNN(dims=[4, 64, 64, 1], key=key_nn)
            value_optimizer = optax.adam(1e-3)
            value_opt_state = value_optimizer.init(eqx.filter(value_net, eqx.is_array))

            optimizer = POLO(
                loss=loss_fn,
                grad_loss=grad_loss_fn,
                lam=config['mppi']['lambda_value'],
                running_cost=running_cost,
                terminal_cost=terminal_cost,
                set_control=set_control,
                mx=mx,
                n_rollouts=N_rollouts,
                sim=config['simulation']['name'],
                baseline=config['mppi']['baseline'],
                sim_traj_mppi_func=None,
                replay_buffer=replay_buffer,
                value_net=value_net,
                value_optimizer=value_optimizer,
                value_opt_state=value_opt_state,
                update_frequency=20,
                mini_batch=20,
                grad_steps=1,
                gamma=1.0
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return optimizer
