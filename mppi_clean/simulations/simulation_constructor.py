import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Tuple, Callable, Dict, Any

class SimulationConstructor:
    @staticmethod
    def create_simulation(sim_name: str, config: Dict[str, Any]) -> Tuple[
        str,  # XML path
        jnp.ndarray,  # qpos_init
        Callable[[Any, jnp.ndarray], Any],  # set_control
        Callable[[Any], float],  # running_cost
        Callable[[Any], float]  # terminal_cost
    ]:
        """Factory method to create simulation components based on simulation name"""
        if sim_name == "cartpole":
            from simulations.cartpole import create_cartpole
            return create_cartpole(config)
        else:
            raise ValueError(f"Unknown simulation: {sim_name}")
