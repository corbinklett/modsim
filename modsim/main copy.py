import numpy as np
from typing import List
from scipy.integrate import odeint

# TODO: enforce types - each state, input, and output has to be a numpy array

class Signal: 
    def __init__(self):
        self.value: float = 0.0  # scalar, enforce type as float
        self.name: str = ""
        self.unit: str = "" 

class SignalSet:
    def __init__(self, signals: List[Signal] = []):
        self.signals: List[Signal] = signals  # Enforce type as List[Signal]
        self.dimension = len(signals)

    def get_vector(self):
        return np.array([signal.value for signal in self.signals])
    
    def set_signal_values(self, vector):
        # place numpy array of values into the signal set
        for i, signal in enumerate(self.signals):
            signal.value = vector[i]

class Subsystem:
    def __init__(self):
        self.inputs = SignalSet()
        self.outputs = SignalSet()
        self.states = SignalSet()
        self.parameters = {}
        # TODO: update subsystem structure to have function primitives in addition to parameters. Could be used for autodiff?
        # The range of types that a parameter could take is wider than what a signal takes, which is just a 'float'

    # TODO: write a function that (optionally) names the states, inputs, and outputs and assigns units
    # def name_variables(self):
        
    def update_outputs(self):
        # TODO: make this easy to write by an AI with Latex input
        pass

    def dynamics(self, time):
        # TODO: make this easy to write by an AI with Latex input
        pass

class SimulationEngine:
    def __init__(self):
        self.subsystems = []
        self.connections = []

    def add_subsystem(self, subsystem):
        self.subsystems.append(subsystem)

    def connect(self, input_signal, output_signal):
        self.connections.append((input_signal, output_signal))

    

    def compute_outputs(self):
        # rearrange connections, if needed, such that outputs can be computed. # some outputs will depend the subsystem's inputs.
        # TODO: throw error if causality is not possible
        subsystem_computed_list = []

        # goal is to rearrange subsystems
        # want to see if a subsystem has no input - if so, compute that subsystem's outputs and put it at the top of the list
        # loop through connections
        # if the connection has an input, 

        # see if the system has any inputs by just looking at that attribute

        # would prefer to know that all inputs are computed first before computing a subsystem's outputs

        # if I have looped through all subsystems and know that 

        # hit a subsystem, see if all inputs (the outputs of these inputs) have been computed (could be multiple inputs from multiple blocks)
        # if so, compute outputs and add to the list

        for subsystem in self.subsystems:
            # find the inputs?

            # LEFT OFF - multi dimension inputs (i.e. if the input to a system is r - y )
            # how can a system's own output be an input?
            # how do I specify which input is u1, u2, etc?
            # change the definition of "connect" slightly

            if len(input_sys.outputs) == subsystem.dim_inputs:
                subsystem.update_outputs()
                subsystem_computed_list.append(subsystem)
                self.subsystems.remove(subsystem
                

        index = 0
        while len(self.subsystems) > 0:
            for connection in self.connections:
                input_sys = connection[0]
                see if it is an output of a subsystem

        index = 0 
        while len(self.subsystems) > 0:
            try:
                self.subsystems[index].update_outputs()
                subsystem_computed_list.append(self.subsystems[index])
                self.subsystems.pop(index)
                index = 0
            except:
                print("ERRORED!")
                index += 1

            # TODO: write error handlign for index exceeding the number of subsystems
                            
        self.subsystems = subsystem_computed_list
        print(self.subsystems)

    def populate_states(self, states):
        # unpack the state vector into the subsystems
        counter = 0
        for subsystem in self.subsystems:
            if subsystem.dim_states > 0:
                subsystem.states = states[counter : counter + subsystem.dim_states]
                counter += subsystem.dim_states

    def dynamics(self, states, time):
        derivative = np.array([])

        # populate states
        self.populate_states(states)

        # compute outputs
        self.compute_outputs()

        for connection in self.connections:
            connection[1].inputs = connection[0].outputs

        for subsystem in self.subsystems:
            val = subsystem.dynamics(time)
            if val is not None:
                derivative = np.append(derivative, val)

        return derivative

    def simulate(self, initial_states, time, dt=0.01):
        t0 = time[0] # tuple
        tf = time[1]
        t = np.arange(t0, tf, dt)
        state_trajectories = odeint(self.dynamics, initial_states, t)
        return state_trajectories


# left off - testing ex2.py with the "Gain" added to see if the response is correct and if the subsystems are rearranged appropriately
    # might not be since np.array([]) * 2.5 is not illegal. This is when doing the "try" on the outputs computation

# left off - next example - assemble a full feedback loop, write test
   