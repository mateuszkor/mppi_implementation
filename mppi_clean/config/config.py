# config/config.py
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    name: str
    path: str

@dataclass
class MPPIConfig:
    n_steps: int
    n_rollouts: int
    lambda_value: float
    initial_control: float

@dataclass
class CostsConfig:
    control_weight: float
    terminal_weight: float

@dataclass
class Config:
    simulation: SimulationConfig
    mppi: MPPIConfig
    costs: CostsConfig

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    simulation_config = SimulationConfig(
        name=config_dict['simulation']['name'],
        path=config_dict['simulation']['path']
    )
    
    mppi_config = MPPIConfig(
        n_steps=config_dict['mppi']['n_steps'],
        n_rollouts=config_dict['mppi']['n_rollouts'],
        lambda_value=config_dict['mppi']['lambda'],
        initial_control=config_dict['mppi']['initial_control']
    )
    
    costs_config = CostsConfig(
        control_weight=config_dict['costs']['control_weight'],
        terminal_weight=config_dict['costs']['terminal_weight']
    )
    
    return Config(
        simulation=simulation_config,
        mppi=mppi_config,
        costs=costs_config
    ), config_dict

def generate_name(config: Config):
    name_elements = [
        config.simulation.name,
        config.mppi.n_steps,
        config.mppi.n_rollouts,
        config.mppi.lambda_value,
        config.mppi.initial_control,
        config.costs.control_weight,
        config.costs.terminal_weight
    ]

    name = "_".join(map(str, name_elements))
    return name