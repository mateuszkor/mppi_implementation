import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Any


def generate_qpos_init(qpos_init_type):
    if qpos_init_type == "random":
        pass
    elif qpos_init_type == "manual":
        pass
    elif qpos_init_type == "zero":
        pass
    else:
        raise ValueError(f"Unknown qpos_init_type: {qpos_init_type}")


def hand_fixed_costs(config: Dict[str, Any]) -> Tuple[
    jnp.ndarray,  # qpos_init
    Callable[[Any, jnp.ndarray], Any],  # set_control
    Callable[[Any], float],  # running_cost
    Callable[[Any], float]  # terminal_cost
]:
    """Create hand fixed simulation components"""
    control_weight = config['costs']['control_weight']
    quat_weight = config['costs']['quat_weight']
    finger_weight = config['costs']['finger_weight']
    terminal_weight = config['costs']['terminal_weight']
    
    def set_control(dx, u):
        forces = u + dx.qpos[:]
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
    def running_cost(dx):
        u = dx.ctrl 
        return float(control_weight) * jnp.sum(u ** 2)
    
    def terminal_cost(dx):
        angle_cost = jnp.sin(dx.qpos[1] / 2) * jnp.pi
        return float(terminal_weight) * (dx.qpos[0] ** 2 + angle_cost ** 2)
    
    return set_control, running_cost, terminal_cost
