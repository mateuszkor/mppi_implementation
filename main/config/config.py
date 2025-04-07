# config/config.py
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import jax.numpy as jnp
import pprint

@dataclass
class SimulationConfig:
    name: str
    path: str
    algo: str
    sensors: bool

    def __post_init__(self):
        if self.sensors:
            path_parts = self.path.split('/')
            if 'shadow_hand' in path_parts and 'shadow_hand_sensors' not in path_parts:
                path_parts[1] = 'shadow_hand_sensors'
                self.path = '/'.join(path_parts)

@dataclass
class MPPIConfig:
    n_steps: int
    n_rollouts: int
    lambda_value: float
    initial_control: float
    baseline: bool
    gamma: float
    td_step: int

@dataclass
class CostsConfig:
    control_weight: float
    terminal_weight: float
    finger_weight: float
    quat_weight: float
    intermediate_weight: float

@dataclass
class HandConfig:
    qpos_init: str
    goal_quat: jnp.ndarray

@dataclass
class NetworkConfig:
    network_dims: list
    update_frequency: int
    mini_batch: int
    grad_steps: int
    net_update_type: str 
    load_model: bool     
    save_model: bool     

@dataclass
class Config:
    simulation: SimulationConfig
    mppi: MPPIConfig
    costs: CostsConfig
    hand: Optional[HandConfig] = None
    network: Optional[NetworkConfig] = None

    def print_config(self):
        # Convert the Config dataclass to a ictionary
        config_dict = asdict(self)
        
        # Pretty print the dictionary
        pprint.pprint(config_dict, indent=4)

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    simulation_config = SimulationConfig(
        name=config_dict['simulation']['name'],
        path=config_dict['simulation']['path'],
        algo=config_dict['simulation']['algo'],
        sensors=config_dict['simulation'].get('sensors', False)
    )
    
    mppi_config = MPPIConfig(
        n_steps=config_dict['mppi']['n_steps'],
        n_rollouts=config_dict['mppi']['n_rollouts'],
        lambda_value=config_dict['mppi']['lambda'],
        initial_control=config_dict['mppi']['initial_control'],
        baseline=config_dict['mppi'].get('baseline', True),
        gamma=config_dict['mppi'].get('gamma', 1.0),
        td_step=config_dict['mppi'].get('td_step', None)
    )
    
    costs_config = CostsConfig(
        control_weight=config_dict['costs']['control_weight'],
        finger_weight=config_dict['costs'].get('finger_weight', None),
        quat_weight=config_dict['costs'].get('quat_weight', None),
        intermediate_weight = config_dict['costs'].get('intermediate_weight', None),
        terminal_weight=config_dict['costs']['terminal_weight']
    )
    
    hand_config = None
    if "hand" in config_dict:
        hand_config = HandConfig(
            qpos_init=config_dict['hand']['qpos_init'],
            goal_quat=jnp.array(config_dict['hand']['goal_quat'])
        )

    network_config = None
    if 'network' in config_dict:
        network_config = NetworkConfig(
            network_dims=config_dict['network']['network_dims'],
            update_frequency=config_dict['network']['update_frequency'],
            mini_batch=config_dict['network']['mini_batch'],
            grad_steps=config_dict['network']['grad_steps'],
            net_update_type=config_dict['network']['net_update_type'],
            load_model=config_dict['network'].get('load_model', False),
            save_model=config_dict['network'].get('save_model', False),
        )
    
    return Config(
        simulation=simulation_config,
        mppi=mppi_config,
        costs=costs_config,
        hand=hand_config,
        network=network_config
    ), config_dict

def generate_name(config_dict: Dict[str, Any]) -> str:
    ''' generates a name for the experiment based on the config.
        Structure of name is: name_usesensors___nsteps_nrollouts_lambda_initialcontrol___controlweight_terminalweight
    '''

    elems = []
    for key, val in config_dict.items():
        for subkey, subval in val.items():
            if subkey == 'path': continue
            elif subkey == 'sensors': 
                if subval: 
                    elems.append('sensor')
                else: 
                    elems.append('nosensor')
            else:
                elems.append(subval)
        if key == 'costs': break
        elems.append("_")

    name = "_".join(map(str, elems))
    return name