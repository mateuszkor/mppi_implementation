import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Any

def cartpole_costs(config: Dict[str, Any]) -> Tuple[
    Callable[[Any, jnp.ndarray], Any],  # set_control
    Callable[[Any], float],  # running_cost
    Callable[[Any], float]  # terminal_cost
]:
    
    """Create cartpole simulation components"""
    path = config['simulation']['path']
    control_weight = config['costs']['control_weight']
    terminal_weight = config['costs']['terminal_weight']
    
    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
    def running_cost(dx):
        u = dx.ctrl 
        return float(control_weight) * jnp.sum(u ** 2)
    
    def terminal_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return float(terminal_weight) * (dx.qpos[0] ** 2 + angle_cost ** 2)
    
    return set_control, running_cost, terminal_cost
