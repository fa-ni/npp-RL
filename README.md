
# Overview
![example workflow](https://github.com/fa-ni/npp-RL/actions/workflows/pytest.yml/badge.svg)
![example workflow](https://github.com/fa-ni/npp-RL/actions/workflows/black.yml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

This repo currently contains the RL Code to learn the npp-implemenation in python from a earlier project. Currently the project which is called "python-backend" is still within this repository. This might be changed later. For more information about the earlier project, please go to section "python-backend".

## RL

### General information & Background
This project uses RL Algorithms from stable-baselines3 to learn Standard Operating Procedure for the nuclear power plant simulation. It builds up a custom OpenAI Gym environment with multiple scenarios and wrappers.
### How to use
TODO: Add how to start training, how to start testing
This project is tested with python version 3.9.
To install the necessary packages there are two options available:
1. Poetry
   1. Install poetry (https://python-poetry.org/)
   2. Optional: Run the command 'poetry config virtualenvs.in-project true' to create a new venv only for this project
   3. Run the command 'poetry install'
2. Requirements.txt
   1. Run the command 'pip install -r requirements.txt'
### Scenarios

### Wrapper

The wrapper classes help to reduce code duplication. The wrappers can be used in a flexible way in the current scenario.
Either no wrapper, one ObservationWrapper, one ActionWrapper or a combination of ActionWrapper and ObservationWrapper
can be used. There is also no need to adjust anything else. With the helper methode make_wrapper, wrapper can be chained
and overhanded to the
'make_vec_env' function from gym.

#### ActionWrappers

The ActionWrappers only modify the number of action parameters/dimensions. ActionSpaceOption2Wrapper is used to be able
to have 3 dimensions in the action space. ActionSpaceOption3Wrapper is used to be able to have 5 dimensions in the
action space. By default, they already can handle all three different scenarios which are included in this package (
continuous,multibinary,multidiscrete). The different action spaces sizes are handled with if statements in all 3
scenarios directly. Which exact parameters are behind these functions can be seen in the description of each class.

#### ObservationWrappers

The ObservationWrappers modify the number of dimensions/parameters for the observation space. As the observation space
is also used in the "reset" and "step" function of the gym environments these functions are also programmed. These
functions (can) get the output of the original functions from the scenarios. ObservationOption2Wrapper has 3 dimensions.
ObservationOption3Wrapper has 6 dimensions. ObservationOption4Wrapper has 5 dimensions. ObservationOption5Wrapper has 10
dimensions. Which exact parameters are behind these functions can be seen in the description of each class.

### Evaluation

To evaluate the performance the eval.py script is used. It computes the mean_reward over multiple test episodes.

### Frontend

### Training

### Logging and model saving

##WHY - Explanations of using different stuff
*Black*: To make a project consistent and easy to read a formatter like black is really helpful.

*Pytest* + Mock: To be able to do unit testing. Mocking is an essential part for unit testing.

*Poetry*: Poetry is a dependency management and resolver tool which also takes into account the transitive dependencies
of packages and therefore allows easy updates with a single command.

*Github Actions*: To make sure that only code with a certain quality is merged to master Github Actions are
used. In every Pull request two Github Actions are run automatically which are running black and pytest to check
if everything is formatted correct and all tests are run successful.

*Pre-Commit*: This tool can check the code before you do a commit from a local device. In this case it is used, so
that no unformatted code can be committed if pr-commit is installed on the local device.

## python-backend
This part is used to simulate a nuclear power plant (npp). It contains the backend logic in python. It uses a similar
logic for the backend as described in [Weyers et al. 2017].

Next to the logic of the npp, this repo contains a function that can start the reactor and keep at stable at a certain
power output.

It also contains main functions to start a Py4J and a Pyjnius Python-Java-Bridge to connect Python with the Java-Program.
With the run_benchmark file, you can create a benchmark of the startup performance of the three different versions ( native
Python, Py4J-Bridge, Pyjnius-Bridge). You can then also analyze the results with the notebook in the analysis folder.
Make sure that you copy the result.csv files to the results folder or change the paths in the notebook.

The python code should later be used to train a reinforcement learning agent to identify whether it is possible to train an
agent to fulfill the task of starting and shutting down a nnp.


## Citations:

Weyers B., Harrison M.D., Bowen J., Dix A., Palanque P. (2017) Case Studies. In: Weyers B., Bowen J., Dix A., Palanque
P. (eds) The Handbook of Formal Methods in Human-Computer Interaction. Human–Computer Interaction Series. Springer,
Cham. https://doi.org/10.1007/978-3-319-51838-1_4
