# Trading agent: trading bitcoin with reinforcement learning
*an INF581 project*

* Report for this project is the file Reinforcement_Learning_Project.pdf


This repository provides an environment based on the gym architecture to implement a trading agent on bitcoin data from 2017 to 2022. It provides multiple state of the art reinforcement learning algoritms (namely [Logistic regression](#Logistic-regression), [Deep Qlearning](#deepqlearning), [A2C](#A2C) and [PolicyGradient](#Policygradient)).

## Table of contents
* [Getting started](#getting-started)
* [Environment](#environment)
* [Training an agent](#training-agent)
* [Agents](#agents)
    * [Logistic agent](#Logistic-agent)
    * [Deepsense DQN agent](#deepsense-dqn-agent)
    * [A2C agent](#A2C-agent)
    * [PolicyGradient agent](#Policygradient)


## Getting started
In order to set up the project, please use a virtual environment that can be created and activated with
```bash
python3 -m venv .venv
source ./.venv/bin/activate
```
Then, install the required libraries with
```bash
pip install --uprgade pip
pip3 install -r requirements.txt
```

## Environment

The game environement is provided as a class `CryptoEnv` located in `src/gym_trading_btc/envs/bitcoin_env.py` which leverages the `gym` based architecture for environments. The dataset used for the bitcoin curved can be found in the `datasets` file and can be modified, to use easier curves, or change the atomicity of the real bitocin curve (minutes, hours). 

The state of the game in intrinsically described by the `config` files located in `src/config_mods` , where *num_actions* represents the actions that an agent is allowed to do, 3 stands for Stay, Sell and Buy, *Window_size* is the number of elements of the past that an agent takes as input to take a decision. Finally, *frame_len* is the number of decision for a single episode.

## Training an agent

Agents can be trained with pre-implemented scripts, that leverages the sinus based method described in the project report. 
Agents are basically trained on easy sinus curves, sums of them and noisy sinus curves, before being trained of the real bitcoin curve. 

To execute the training pipeline on Deepsense Double Deep Q learning, run:
```bash
bash train_pipeline-sh/train_pipeline_deespsens.sh
```

Training takes approx. 8 hours on Quadro 4000 GPU.
Figures to described progress during training are gradually saved in the `figs`file.


## Agent
A precumputed agent can be run with

```bash
python main.py

optional arguments:
    --config
                To precise which agent to use, load it and its config file. Options are 'dqn_base', 'dqn_deepsense', 'config_a2c' and 'classifier' If no argument is given, A2C agent is loaded
    --df_name
                To manually set the the dataframe to use for cypto data in the environment
    --save_path
                Save path during training
    --load_path
                Load path for precomputed model
    --num_episode
                Load
    --save
                Boolean, 0 for False and 1 for True
    --load
                Boolean, 0 for False and 1 for True
    --lr
                Manually set the learning rate for an episode
    --config
                To precise which agent to use, load it and its config file. Options are 'dqn_base', 'dqn_deepsense', 'config_a2c' and 'classifier'
    --classifier_model
    --classifier_objective
```

These arguments overrite config arguments that can be modified in the `config_mods`file