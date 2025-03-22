import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Tuple, Callable, Dict, Any

class SimulationConstructor:
    @staticmethod
    def create_simulation(sim_name: str, config: Dict[str, Any], mx: mjx.Data, key) -> Tuple[
        jnp.ndarray,  # qpos_init
        Callable[[jnp.ndarray, float], bool],
        Callable[[Any], Dict[str, Any]],
        Callable[[Any, jnp.ndarray], Any],  # set_control
        Callable[[Any], float],  # running_cost
        Callable[[Any], float]  # terminal_cost
    ]:
        """Factory method to create simulation components based on simulation name"""
        if sim_name == "cartpole":
            from simulations.cartpole import cartpole_costs, termination_function, get_log_data
            qpos_init = jnp.array([0.0, 3.14])
            return qpos_init, termination_function, get_log_data, *cartpole_costs(config)
        elif sim_name == "hand_fixed":
            from simulations.hand_fixed import generate_qpos_init, hand_fixed_costs, termination_function, get_log_data 
            qpos_init = generate_qpos_init(key, config['hand'], mx)
            return qpos_init, termination_function, get_log_data, *hand_fixed_costs(config)
        elif sim_name == "hand_free":
            from simulations.hand_free import generate_qpos_init, hand_free_costs, termination_function, get_log_data
            qpos_init = generate_qpos_init(config['hand'], mx)
            return qpos_init, termination_function, get_log_data, *hand_free_costs(config)
        else:
            raise ValueError(f"Unknown simulation: {sim_name}")
