import numpy as np
from typing import Any, List
from scipy.integrate import odeint

# TODO: enforce types - each state, input, and output has to be a numpy array

class Signal: 
    def __init__(self):
        self._value: float = 0.0  # scalar, enforce type as float
        self.name: str = ""
        self.unit: str = ""
        self.count: int = 0 # keeps track of assignments to this signal

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self.count += 1
        

class SignalSet:
    def __init__(self, dimension):
        self.signals: List[Signal] = [Signal() for _ in range(dimension)]
        self.dimension = len(dimension)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self, key
        elif isinstance(key, str):
            for signal in self.signals:
                if signal.name == key:
                    return signal
        else:
            raise KeyError("Invalid key type")

    def names_units(self, names, units=None):
        # TODO: add similar function to Subsystem like "name_outputs" and "name_inputs" such that you can't have an output named the same as an input
        if len(names) != len(self.signals):
            raise ValueError("Number of names must match the number of signals")
        if units is not None and len(units) != len(self.signals):
            raise ValueError("Number of units must match the number of signals")

        for i, signal in enumerate(self.signals):
            signal.name = names[i]
            if units is not None:
                signal.unit = units[i]

    def get_vector(self):
        return np.array([signal.value for signal in self.signals])
    
    def set_values(self, vector):
        if np.isscalar(vector):
            vector = np.array([vector])
        # place numpy array of values into the signal set
        for i, signal in enumerate(self.signals):
            signal.value = vector[i]

class Subsystem:
    def __init__(self, dim_states, dim_inputs, dim_outputs, parameters={}):
        self.states = SignalSet(dim_states)
        self.inputs = SignalSet(dim_inputs)
        self.outputs = SignalSet(dim_outputs)
        self.parameters = parameters

        # TODO: update subsystem structure to have function primitives in addition to parameters. Could be used for autodiff?
        # The range of types that a parameter could take is wider than what a signal takes, which is just a 'float'

    # TODO: write a function that (optionally) names the states, inputs, and outputs and assigns units
    # def name_variables(self):
        
    def __getitem__(self, key):
        return self, key
        
    def update_system(self, time):
        x = self.states.get_vector()
        u = self.inputs.get_vector()

        if len(x) == 0:
            # system is memoryless and output depends only on input
            y = self.update_outputs(self, u, time)
        else: 
            # pass the state into the output function
            # TODO: allow outputs that are a function of state and feedthrough input 
            # (or, rather, reconstruct system to seperate the feedforward component, assuming x & u are coupled)
            y = self.update_outputs(self, x, time)
        
        self.outputs.set_values(y)
        xdot = self.dynamics(self, x, u, time)
        return xdot
        
    def update_outputs(self, state_or_input, time):
        pass

    def dynamics(self, x, u, time):
        pass

class SimulationEngine:
    def __init__(self):
        self.subsystems = [] # list of Subystem objects
        self.connections = [] # list of dictionaries
        self.count = 0

    def add_subsystem(self, subsystem):
        self.subsystems.append(subsystem)

    def connect(self, output_tuple, input_tuple):
        # output signal -> input signal
        #  self.connections is a list of subsystems and their connections

        output_sys = output_tuple[0]
        output_signal_name = output_tuple[1]
        output_signal = output_sys.outputs[output_signal_name]

        input_sys = input_tuple[0]
        input_signal_name = input_tuple[1]
        input_signal = input_sys.inputs[input_signal_name]

        # check to see if subsystem already has connections
        system_found = False
        for connection in self.connections:
            if connection['system'] == input_sys:
                connection['connections'].append((output_signal, input_signal))
                system_found = True
                break
        
        if not system_found:
            self.connections.append({'system': input_sys, 'connections': [(output_signal, input_signal)]})
        
    def update_systems(self, time):

        subsystem_computed_list = []
    
        # it's going to have to find all the connectins that belong toa  subsystem

        # will need to re-assign the incoming signal value when the count = the upstream output count
        for connection in self.connections:
            if connection[0].count == connection[1].count + 1:
                connection[1].value = connection[0].value 
            else:


            LEFT OFF - trying to manage connections, possibly sorting by subsystem
            # only do the single subsystem's connections, not all connections arbitrarily
            # when looping through connections for a subsystem again, increment the count of each?
            # but that could eff you up if one subsystem is connected to multiple other subsystems
        index = 0

        IDEA - first compute subsystems that have no feedforward

        
        while len(self.subsystems) > 0:

            HERE TOO
            # if subsystem has no states, then see if all inputs have been updated appropriately
            # look at all incoming connections in connections object and see if the count is < 1 to the output count
            # if not, then the output will need to put somewhere and updated again?

            connections[self.subsystems[index]].update()


            self.subsystems[index]
            try:
                xdot = self.subsystems[index].update_system(time)
            except: 
                index += 1 


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
                
    def populate_states(self, states):
        # unpack the state vector into the subsystems
        counter = 0
        for subsystem in self.subsystems:
            if len(subsystem.states) > 0:
                subsystem.states.set_values(states[counter : counter + len(subsystem.states)])
                counter += len(subsystem.states)

    def dynamics(self, states, time):
        derivative = np.array([])

        # populate states
        self.populate_states(states)

        # compute outputs
        self.update_systems()

        for subsystem in self.subsystems:
            val = subsystem.dynamics(time)
            if val is not None:
                derivative = np.append(derivative, val)

        return derivative

    def simulate(self, initial_states, time, dt=0.01):
    
    # TODO: loop through subsystems and assign counts to zero
        self.count = 0 # reset the function call count to help enforce causility
        t0 = time[0] # tuple
        tf = time[1]
        t = np.arange(t0, tf, dt)
        state_trajectories = odeint(self.dynamics, initial_states, t)
        return state_trajectories


# left off - testing ex2.py with the "Gain" added to see if the response is correct and if the subsystems are rearranged appropriately
    # might not be since np.array([]) * 2.5 is not illegal. This is when doing the "try" on the outputs computation

# left off - next example - assemble a full feedback loop, write test
   