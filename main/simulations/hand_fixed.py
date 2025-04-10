import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Any

def quaterion_diff(q1,q2):
    q1_norm = q1 / jnp.linalg.norm(q1)
    q1_conj = jnp.array([q1_norm[0], -q1_norm[1], -q1_norm[2], -q1_norm[3]])
    s1, x1, y1, z1 = q1_conj
    s2, x2, y2, z2 = q2
    return jnp.array([
        s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2, 
        s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2,  
        s1 * y2 - x1 * z2 + y1 * s2 + z1 * x2,  
        s1 * z2 + x1 * y2 - y1 * x2 + z1 * s2   
    ])

def calculate_angular_distance(q1, q2):
    quat_diff = quaterion_diff(q1, q2)
    angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
    return angle

def termination_function(qpos, epsilon, print_enabled):
    ball_quat, goal_quat = qpos[24:28], qpos[28:32]
    angle = calculate_angular_distance(ball_quat, goal_quat)
    angle = jnp.degrees(angle)
    if print_enabled:
        print(f"Current ball_quat={ball_quat}")
        print(f"Remaining angle to goal position = {angle}")

    return jnp.less(angle, epsilon)

def get_log_data(separate_costs, optimal_cost, step, qpos):
    ctrl_cost, quat_cost, finger_cost, running_cost, final_cost = separate_costs
    ball_quat, goal_quat = qpos[24:28], qpos[28:32]
    
    angle = calculate_angular_distance(ball_quat, goal_quat)
    log_data = {"Optimal cost": optimal_cost, 
            "Control cost": ctrl_cost,
            "Ball orientation cost": quat_cost,
            "Finger sensor cost": finger_cost,
            "Running cost": running_cost, 
            "Terminal cost": final_cost, 
            "Remaining angle": jnp.degrees(angle),
            "Step": step}
    return log_data

def generate_qpos_init(key, config_hand, mx):
    qpos_init_type = config_hand['qpos_init']
    goal_quat = config_hand['goal_quat']

    if qpos_init_type == "default":
        qpos_init = mx.qpos0
        qpos_init = jnp.array([
            0.0041, -0.28, 6.2e-18, -0.026, -0.0007, -0.00029, 
            6e-18, -0.026, -0.0007, -0.00029, -1.2e-06, -0.026, 
            -0.0007, -0.00029, -0.0043, -9e-05, -0.026, -0.0007, 
            -0.00029, -9.7e-05, -0.0016, -0.021, 0.0014, 0.00047, 
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0
        ])
    elif qpos_init_type == "manual":
        qpos_init = jnp.array([
            -0.03,   -0.36,   -0.35,    0.82,    0.89,    0.61,   
            -0.021,   1.1,    0.41,    0.88,   -0.074,   0.67,    
            1.4,   -0.0024,   0.43,   -0.34,    0.54,    0.9,     
            0.39,   0,   1.2,    0.089,    0.25,     0.6,    
            1.0, 0.0, 0.0, 0.0,    
            1.,     0.,      0.,      0.
        ])
    elif qpos_init_type == "random":
        qpos_init = jax.random.uniform(key, mx.nq, minval=-0.1, maxval=0.1)
    else:
        raise ValueError(f"Unknown qpos_init_type: {qpos_init}")
    
    qpos_init = qpos_init.at[24:32].set(mx.qpos0[24:32])
    qpos_init = qpos_init.at[28:32].set(goal_quat)

    angle = calculate_angular_distance(qpos_init[24:28], goal_quat)
    print(f"Initial ball_quat={qpos_init[24:28]}")
    print(f"Remaining angle to goal position = {jnp.degrees(angle)}")
    return qpos_init

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
    goal_quat = config['hand']['goal_quat']
    use_sensors = config['simulation']['sensors']
    
    def set_control(dx, u):
        forces = u + dx.qpos[:24]
        return dx.replace(ctrl=dx.ctrl.at[:].set(forces))

    def running_cost(dx, optimal=False):
        u = dx.ctrl
        ctrl_cost = float(control_weight) * jnp.sum(u ** 2)

        ball_quat = dx.qpos[24:28]
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        quat_cost = float(quat_weight) * (angle ** 2)
        # # jax.debug.print("quat_cost: {x}", x=quat_cost)

        finger_cost = 0.0
        if use_sensors:
            thumb_sensor_data = dx.sensordata[0]
            thumb_contact_cost = (1/(thumb_sensor_data + (1/100)))
            # jax.debug.print("thumb_cost: {x}", x = thumb_contact_cost)

            finger_data, palm_data = dx.sensordata[1:5], dx.sensordata[5]
            finger_cost = (jnp.sum(1/(finger_data + (1/25))) + thumb_contact_cost) * float(finger_weight)

        return ctrl_cost, quat_cost, finger_cost

    def terminal_cost(dx):
        # -------- ball quats -----------
        ball_quat = dx.qpos[24:28]
        quat_diff = quaterion_diff(ball_quat, goal_quat)
        angle = 2 * jnp.arccos(jnp.abs(quat_diff[0])) 
        quat_cost = float(quat_weight) * (angle ** 2)

        # jax.debug.print("quat_cost: {x}", x=quat_cost)
        return quat_cost * float(terminal_weight)


    return set_control, running_cost, terminal_cost
