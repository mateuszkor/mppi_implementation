import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Any


def termination_function(qpos, epsilon: float, print_enabled: bool) -> bool:
    # angle = jnp.mod(qpos[1], jnp.pi)
    mod_angle = jnp.mod(qpos[1], 2 * jnp.pi)
    rem_angle = jnp.degrees(jnp.pi - jnp.abs(mod_angle - jnp.pi)) 
    if print_enabled:
        print(f"Current qpos={qpos}")
        print(f'Remaining angle to goal position = {rem_angle}')

    return jnp.less(rem_angle, epsilon)


def get_log_data(separate_costs, optimal_cost, step, qpos):
    running_cost, final_cost = separate_costs
    mod_angle = jnp.mod(qpos[1], 2 * jnp.pi)
    rem_angle = jnp.pi - jnp.abs(mod_angle - jnp.pi) 
    log_data = {"Optimal Cost": optimal_cost.astype(float), 
                "Running Cost": running_cost.astype(float), 
                "Terminal Cost": final_cost.astype(float), 
                "Angle": jnp.degrees(rem_angle).astype(float),
                "Step": step}
    return log_data


def cartpole_costs(config: Dict[str, Any]) -> Tuple[
    Callable[[Any, jnp.ndarray], Any],  # set_control
    Callable[[Any], float],  # running_cost
    Callable[[Any], float]  # terminal_cost
]:
    
    """Create cartpole simulation components"""
    algorithm = config['simulation']['algo']
    control_weight = config['costs']['control_weight']
    intermediate_weight = config['costs']['intermediate_weight']
    terminal_weight = config['costs']['terminal_weight']
    
    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
    def running_cost(dx):
        running_cost = jnp.sum(dx.ctrl  ** 2)
        positional_cost = 0.0
        if algorithm == "polo" or algorithm == "polo_td":
            angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi 
            positional_cost = (dx.qpos[0] ** 2 + angle_cost ** 2) * float(intermediate_weight)

        return float(control_weight) * running_cost + positional_cost
    
    def terminal_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return float(terminal_weight) * (dx.qpos[0] ** 2 + angle_cost ** 2)
    
    return set_control, running_cost, terminal_cost
