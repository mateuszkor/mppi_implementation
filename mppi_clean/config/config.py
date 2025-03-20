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
    sensors: bool

@dataclass
class MPPIConfig:
    n_steps: int
    n_rollouts: int
    lambda_value: float
    initial_control: float
    baseline: bool

@dataclass
class CostsConfig:
    control_weight: float
    terminal_weight: float
    finger_weight: float
    quat_weight: float

@dataclass
class HandConfig:
    qpos_init: str
    goal_quat: jnp.ndarray

@dataclass
class Config:
    simulation: SimulationConfig
    mppi: MPPIConfig
    costs: CostsConfig
    hand: Optional[HandConfig] = None

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
        sensors=config_dict['simulation'].get('sensors', True)
    )
    
    mppi_config = MPPIConfig(
        n_steps=config_dict['mppi']['n_steps'],
        n_rollouts=config_dict['mppi']['n_rollouts'],
        lambda_value=config_dict['mppi']['lambda'],
        initial_control=config_dict['mppi']['initial_control'],
        baseline=config_dict['mppi'].get('baseline', True)
    )
    
    costs_config = CostsConfig(
        control_weight=config_dict['costs']['control_weight'],
        finger_weight=config_dict['costs'].get('finger_weight', None),
        quat_weight=config_dict['costs'].get('quat_weight', None),
        terminal_weight=config_dict['costs']['terminal_weight']
    )
    
    hand_config = None
    if "hand" in config_dict:
        hand_config = HandConfig(
            qpos_init=config_dict['hand']['qpos_init'],
            goal_quat=jnp.array(config_dict['hand']['goal_quat'])
        )
    
    return Config(
        simulation=simulation_config,
        mppi=mppi_config,
        costs=costs_config,
        hand=hand_config
    ), config_dict

def generate_name(config_dict: Dict[str, Any]) -> str:
    ''' generates a name for the experiment based on the config.
        Structure of name is: name_usesensors___nsteps_nrollouts_lambda_initialcontrol___controlweight_terminalweight
    '''

    elems = []
    for key, val in config_dict.items():
        for subkey, subval in val.items():
            if subkey == 'path': continue
            elems.append(subval)
        if key == 'costs': break
        elems.append("_")

    name = "_".join(map(str, elems))
    return name