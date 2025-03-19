import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Tuple, Callable, Dict, Any

class SimulationConstructor:
    @staticmethod
    def create_simulation(sim_name: str, config: Dict[str, Any]) -> Tuple[
        jnp.ndarray,  # qpos_init
        Callable[[Any, jnp.ndarray], Any],  # set_control
        Callable[[Any], float],  # running_cost
        Callable[[Any], float]  # terminal_cost
    ]:
        """Factory method to create simulation components based on simulation name"""
        if sim_name == "cartpole":
            from simulations.cartpole import cartpole_costs
            qpos_init = jnp.array([0.0, 3.14])
            return qpos_init, cartpole_costs(config)
        elif sim_name == "hand_fixed":
            from simulations.hand_fixed import generate_qpos_init, hand_fixed_costs
            return generate_qpos_init(config), hand_fixed_costs(config)
        elif sim_name == "hand_free":
            pass
        else:
            raise ValueError(f"Unknown simulation: {sim_name}")
