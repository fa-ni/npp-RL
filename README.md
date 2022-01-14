# python-backend

This repo is used to simulate a nuclear power plant (npp). It contains the backend logic in python. It uses a similar
logic for the backend as described in [Weyers et al. 2017].

Next to the logic of the npp, this repo contains a function that can start the reactor and keep at stable at a certain
power output.

It also contains main functions to start a Py4J and a Pyjnius Python-Java-Bridge to connect Python with the Java-Program.
With the run_benchmark file, you can create a benchmark of the startup performance of the three different versions ( native
Python, Py4J-Bridge, Pyjnius-Bridge). You can then also analyze the results with the notebook in the analysis folder.
Make sure that you copy the result.csv files to the results folder or change the paths in the notebook.

The python code should later be used to train a reinforcement learning agent to identify whether it is possible to train an
agent to fulfill the task of starting and shutting down a nnp. The detailed task to solve, the code and the description
for the reinforcement approach will be in this (TODO) repository.

# How to run:

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
