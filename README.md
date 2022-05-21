
# Overview
![example workflow](https://github.com/fa-ni/npp-RL/actions/workflows/pytest.yml/badge.svg)
![example workflow](https://github.com/fa-ni/npp-RL/actions/workflows/black.yml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

This repo contains  code of a python npp simulation (from an earlier project) and code which can be used to train different reinforcement learning agents.
The base simulation was implemented in java from Weyers (see below). Thank you for letting us use the original java simulation of the npp.
It also contains code and jupyter notebooks for an in depth analysis of the results from the reinforcement learning agents.
For more information of the npp simulation itself please find the readme under src and an overview at the end of this readme.
The information about the reinforcement learning and the analysis can be found in this file down below.

## RL

This project uses RL algorithms from the stable-baselines3 implementation to learn a standard operating procedure (sop) for the npp simulation.
It builds up a custom OpenAI Gym environment with multiple scenarios and wrappers.
### How to use
#### Setup
This project is tested with python version 3.9.
To install the necessary packages there are two options available:
1. Poetry
   1. Install poetry (https://python-poetry.org/)
   2. Optional: Run the command 'poetry config virtualenvs.in-project true' to create a new venv only for this project
   3. Run the command 'poetry install'
2. Requirements.txt
   1. Run the command 'pip install -r requirements.txt'

#### Start RL Training of Agents
The main method is used to start the training of rl agents. You can just configure which environment,
which wrappers and which algorithms to use. Everything else like saving the best model of a training run,
saving tensorboard logs etc. is done automatically.

#### Use and evaluate a RL Agent
There are two option to use a RL agent.
1. You can use the eval.py script to get the basic information from your agent like return, observations within the run etc..
2. You can use the eval_frontend.py script if you also have the suitable jar file here to let your
   agent execute the action over the frontend, so that you can actually see what is happening. However, this
   is not possible with all agents. Currently, only agents which use the action_wrapper_option3 are
   supported.
#### Evaluation and detailed analysis of lots of models / already trained agents
You can use the jupyter notebooks provided on the top folder level.
Maybe you need to adjust paths to point to the right models. The phase2 jupyter notebook
contains general analytics about all trained models. The phase3 is used to get a
detailed analysis of some chosen models.


### Information about the implementation of the environments and wrappers

#### Scenarios
There are three different scenarios implemented.
1. Uses box action space which means only continues action are allowed.
2. Uses a binary action space which means all actions are transformed to binary decisions.
3. Uses a multidiscrete action space which means that there are multiple discrete option per action dimension which the
   agent can use at each timestep.

Each environment has a default of 250 timesteps per episode. This can be overwritten by
passing a parameter to the specific environments.

Each environment starts the simulation at ground zero. This can be overwritten by
passing a parameter to the specific environments.


#### Wrapper

The wrapper classes help to reduce code duplication. The wrappers can be used in a flexible way in the current scenario.
No wrapper can be used a single one or multiple different ones together. There is also no need to adjust anything else.
With the helper methode make_wrapper, wrapper can be chained
and overhanded to the 'make_vec_env' function from gym.

##### ActionWrappers

The ActionWrappers only modify the number of action parameters/dimensions. ActionSpaceOption2Wrapper is used to be able
to have 3 dimensions in the action space. ActionSpaceOption3Wrapper is used to be able to have 5 dimensions in the
action space. By default, they already can handle all three different scenarios which are included in this package (
continuous, multibinary, multidiscrete). The different action spaces sizes are handled with if statements in all 3
scenarios directly.

##### ObservationWrappers

The ObservationWrappers modify the number of dimensions/parameters for the observation space. As the observation space
is also used in the "reset" and "step" function of the gym environments these functions are also programmed. These
functions (can) get the output of the original functions from the scenarios. ObservationOption2Wrapper has 3 dimensions.
ObservationOption3Wrapper has 7 dimensions. ObservationOption4Wrapper has 6 dimensions. ObservationOption5Wrapper has 11
dimensions.

##### NPPAutomationWrapper
If this wrapper is used another part in the backend logic of the npp simulation is used.
This changes the behaviour of the simulation.

##### RewardWrapper
The reward wrappers are being used to choose different reward functions to train and evaluate
the agent with.

##### More Wrappers

There are more wrappers which are being used for some experiments of the trained agents. However,
they could also be used during the training. These noise wrappers are adding different types of
noise to the observations and the backend logic.

###Jupyter Notebooks

There are some jupyter notebooks on the top level of this project. They are used to analyse the trained agents.
First they will evaluate a lot of different stuff and then save the results as a csv file. This csv file is then
used in the jupyter notebooks to analyze the evaluation results. A lot of different plots are produced. Different tables
are created and in general pandas and scipy are used to analyze all the produced data.

##WHY - Explanations of using different stuff
*Black*: To make a project consistent and easy to read a formatter like black is really helpful.

*Pytest* + Mock: To be able to do unit testing. Mocking is an essential part for unit testing.

*Poetry*: Poetry is a dependency management and resolver tool which also takes into account the transitive dependencies
of packages and therefore allows easy updates with a single command.

*Github Actions*: To make sure that only code with a certain quality is merged to master Github Actions are
used. In every Pull request two Github Actions are run automatically which are running black and pytest to check
if everything is formatted correct and all tests are run successful.

*Pre-Commit*: This tool can check the code before you do a commit from a local device. In this case it is used, so
that no unformatted code can be committed if pre-commit is installed on the local device.

## npp simulation
This part is used to simulate a nuclear power plant (npp). It contains the backend logic in python. It uses a similar
logic for the backend as described in [Weyers et al. 2017].

Next to the logic of the npp, this repo contains a function that can start the reactor and keep it stable at a certain
power output.

(The following part ist not relevant for the master thesis):
It also contains main functions to start a Py4J and a Pyjnius Python-Java-Bridge to connect Python with the Java-Program.
With the run_benchmark file, you can create a benchmark of the startup performance of the three different versions (native
Python, Py4J-Bridge, Pyjnius-Bridge). You can then also analyze the results with the notebook in the analysis folder.
Make sure that you copy the result.csv files to the results folder or change the paths in the notebook.

The python code should later be used to train a reinforcement learning agent to identify whether it is possible to train an
agent to fulfill the task of starting and shutting down a nnp.


## Citations:

Weyers B., Harrison M.D., Bowen J., Dix A., Palanque P. (2017) Case Studies. In: Weyers B., Bowen J., Dix A., Palanque
P. (eds) The Handbook of Formal Methods in Human-Computer Interaction. Human–Computer Interaction Series. Springer,
Cham. https://doi.org/10.1007/978-3-319-51838-1_4
