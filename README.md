# MPPI and POLO Implementation

A modular JAX-based implementation of **Model Predictive Path Integral (MPPI)** and **POLO (Plan Online, Learn Offline)** algorithms for continuous control tasks using MuJoCo.

## 📦 Installation

Clone the repo and install dependencies with Pipenv:

```bash
git clone https://github.com/mateuszkor/mppi_implementation.git
cd mppi_implementation
pipenv install
pipenv shell
```

Make sure you have `mujoco` and its dependencies properly set up on your system.

---

## 🧠 Algorithms

You can run one of the following algorithms:

- `mppi`: Standard Model Predictive Path Integral control.
- `polo`: POLO with trajectory optimization and value function learning.
- `polo_td`: POLO with temporal-difference value learning.

---

## 🧪 Tasks

Available simulation environments:

- `swingup`
- `hand_fixed`
- `hand_free`

Each task has its own configuration file under `config/{algorithm}/`.

---

## 🛠️ Usage

To run a simulation:

```bash
python runner.py
```

If running on MAC use following:
```bash
mjpython runner.py
```

You can modify the algorithm and task at the bottom of `runner.py`:

```python
algorithm = "vanilla_mppi"     # Options: "vanill_mppi", "polo", "polo_td"
simulation = "swingup"  # Options: "swingup", "hand_fixed", "hand_free"
```

Configurations are loaded from:

```
config/{algorithm}/{task}.yaml
```

Logging to Weights & Biases is optional and can be toggled with:

```python
use_wandb = 1
```

---

## 📁 Project Structure

```
├── runner.py                  # Entry point for running simulations
├── config/                  # YAML configs for algorithms/tasks
├── models/                  # MPPI, POLO implementations
├── nn/                      # Neural network modules
├── simulations/             # Simulation constructors and cost functions
├── utils/                   # Replay buffer, helpers
```

---

## 📈 Logging with W&B (optional)

To enable Weights & Biases tracking:

1. Set `use_wandb = 1` in `runner.py`.
2. Add your API key or configure W&B locally.