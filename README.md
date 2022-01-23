
# Overview
This repo currently contains the RL Code to learn the npp-implemenation in python from a earlier project. Currently the project which is called "python-backend" is still within this repository. This might be changed later. For more information about the earlier project, please go to section "python-backend".

## RL

### General information & Background
This project uses RL Algorithms from stable-baselines3 to learn Standard Operating Procedure for the nuclear power plant simulation. It builds up a custom OpenAI Gym environment with multiple scenarios and wrappers.
### How to use

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

### How to run:

To run the benchmark install the packages from the requirements.txt via ```pip install -r requirements.txt```.  You need to make sure that you
have the NPP_Simu.jar within the main folder. This jar contains the compiled NPP_Simu program from this repository https://github.com/npp-masterthesis/npp-java-adjusted with all its dependencies from the lib folder.
Also make sure that you have installed the Java JDK. This program was tested with OpenJDK 11 and Python 3.8 on a Windows 10 machine.
Under the raspberry pi OS it also runs under Python 3.7.

You can then execute the run_benchmark.sh file. This will produce three different csv files with the results.
If you execute this script under a linux environment you might need to change
the script to use "python3" instead of "python". You can also execute the main functions itself. For that you need to pass in a ```--round-number``` argument
which is used in the benchmark to understand which iteration it was.

If you also want to use the analysis folder which is used to analyze the results. You also need to install scipy,
matplotlib, numpy and jupyter notebooks.

## Citations:

Weyers B., Harrison M.D., Bowen J., Dix A., Palanque P. (2017) Case Studies. In: Weyers B., Bowen J., Dix A., Palanque
P. (eds) The Handbook of Formal Methods in Human-Computer Interaction. Human–Computer Interaction Series. Springer,
Cham. https://doi.org/10.1007/978-3-319-51838-1_4
