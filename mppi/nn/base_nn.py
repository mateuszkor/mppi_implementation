import equinox as eqx
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

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
