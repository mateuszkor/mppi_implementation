import equinox as eqx
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import os

class Network(eqx.Module, ABC):
    """
    Abstract base class for policies. Users should inherit from this class
    and implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, x, t):
        """
        Forward pass of the network.

        Args:
            x (jnp.ndarray): Input features.
            t (jnp.ndarray): Additional temporal or contextual information.

        Returns:
            jnp.ndarray: Network output.
        """
        pass

    @staticmethod
    @eqx.filter_jit
    def make_step(dxs, optim, model, state, ctx, user_key):
        """
        Performs a single optimization step.

        Args:
            dxs: ..
            optim: Optimizer instance (e.g., from optax).
            model (BasePolicy): The model to update.
            state: Optimizer state.
            ctx: Context object containing additional information like loss function.

        Returns:
            Tuple[BasePolicy, state, float]: Updated model, updated state, and loss value.
        """
        params, static = eqx.partition(model, eqx.is_array)
        (loss_value, res), grads = jax.value_and_grad(ctx.cbs.loss_func, has_aux=True)(
            params, static, dxs, ctx, user_key
        )
        # res
        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)

        return model, state, loss_value, res

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

# Function to save the value function and optimizer state
def save_model(value_net, value_opt_state, save_path="saved_models/value_function.eqx"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving value function to {save_path}")
    eqx.tree_serialise_leaves(save_path, (value_net, value_opt_state))

def load_model(load_path, like_model, like_opt_state):
    print(f"Loading value function from {load_path}")
    value_net, value_opt_state = eqx.tree_deserialise_leaves(load_path, like=(like_model, like_opt_state))
    return value_net, value_opt_state