import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple, Dict, Any
from models.mppi import MPPI
from models.polo import POLO
from utils.replay_buffer import ReplayBuffer
from nn.base_nn import ValueNN, load_model
import os

class OptimizerConstructor:
    @staticmethod
    def create_optimizer(
        algorithm: str,
        config,
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

        if algorithm == "vanilla_mppi":
            print("INITIALISING vanilla_mppi")
            optimizer = MPPI(
                loss=loss_fn,
                grad_loss=grad_loss_fn,
                lam=config.mppi.lambda_value,
                running_cost=running_cost,
                terminal_cost=terminal_cost,
                set_control=set_control,
                mx=mx,
                n_rollouts=config.mppi.n_rollouts,
                sim=config.simulation.name,
                baseline=config.mppi.baseline,
                sim_traj_mppi_func=None,
            )
        elif algorithm == "polo":
            print("INITIALISING POLO")
            key, key_nn = jax.random.split(key)
            replay_buffer = ReplayBuffer(capacity=100000)
            
            value_net = ValueNN(dims=config.network.network_dims, key=key_nn)
            value_optimizer = optax.adam(1e-3)
            value_opt_state = value_optimizer.init(eqx.filter(value_net, eqx.is_array))

            optimizer = POLO(
                loss=loss_fn,
                grad_loss=grad_loss_fn,
                lam=config.mppi.lambda_value,
                running_cost=running_cost,
                terminal_cost=terminal_cost,
                set_control=set_control,
                mx=mx,
                n_rollouts=config.mppi.n_rollouts,
                sim=config.simulation.name,
                baseline=config.mppi.baseline,
                sim_traj_mppi_func=None,
                replay_buffer=replay_buffer,
                value_net=value_net,
                value_optimizer=value_optimizer,
                value_opt_state=value_opt_state,
                update_frequency=config.network.update_frequency,
                mini_batch=config.network.mini_batch,
                grad_steps=config.network.grad_steps,
                gamma=config.mppi.gamma,
                net_update_type=config.network.net_update_type
            )

            # Load the saved model if it exists
            load_path = f"saved_models/value_function_{config.simulation.name}.eqx"  # Change to the correct path
            if os.path.exists(load_path) and config.network.load_model:
                print("Loading pre-trained model...")
                optimizer.value_net, optimizer.value_opt_state = load_model(load_path, value_net, value_opt_state)
            else:
                print("No saved model found. Starting from scratch.")

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return optimizer
