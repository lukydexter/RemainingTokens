# RemainingTokens
Repository with the code for the submitted paper:
> **Revisiting Token-Based Fitness for Streaming Conformance Checking**

Based on the master's thesis by **Mihály Tass**. This repository contains the software artifacts implemented during the work. 
> **Real-Time Cyberattack Detection Using Streaming Process Mining Techniques**
(https://github.com/tassmihaly/compass-attack-detection/tree/master)


## Project Structure

- `src/` - core framework: Petri net models, conformance-checking algorithms, real-time detection, monitoring, and visualization
- `config/` - runtime configuration files

## Installation

```powershell
pip install -e ".[dev]"
```

## Running the Evaluation Pipeline on the Dataset (https://doi.org/10.5281/zenodo.1194057.)

This runs the evaluation on the dataset and provides the results. It needs the dataset downloaded inside a folder named `Tests_online`

```powershell
python python token_replay_new_tests.py
```
Produces the results that are analyzed with: 

```powershell
python python analyze_results.py
```
to create all the plots and results presented in the paper


## Running on the Stress test (https://doi.org/10.5281/zenodo.1194057.)

```powershell
python stress_test.py
```

Produces the results. Also need the dataset inside a folder named `Tests_online` 

```powershell
python stress_test_analysis.py
```

On the results of stress_test.py allows to create all the plots and results presented in the paper
