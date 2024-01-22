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
        self.dimension = dimension

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.signals[key]
        elif isinstance(key, str):
            for signal in self.signals:
                if signal.name == key:
                    return signal
        else:
            raise KeyError("Invalid key type")
        
    def __len__(self):
        return self.dimension

    def names_units(self, names, units=None):
        # TODO: add similar function to Subsystem like "name_outputs" and "name_inputs" such that you can't have an output named the same as an input
        if isinstance(names, str):
            names = [names]  # Convert the string into a list with a single element
        elif not isinstance(names, list):
            raise ValueError("Names must be a string or a list")
        
        if isinstance(units, str):
            units = [units]  # Convert the string into a list with a single element
        elif not isinstance(units, list) and units is not None:
            raise ValueError("Units must be a string or a list")

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
        # TODO: add name attribute

        # TODO: update subsystem structure to have function primitives in addition to parameters. Could be used for autodiff?
        # The range of types that a parameter could take is wider than what a signal takes, which is just a 'float'

    # TODO: write a function that (optionally) names the states, inputs, and outputs and assigns units
    # def name_variables(self):
        
    def __getitem__(self, key):
        return self, key
        
    def update_outputs(self, time):
        x = self.states.get_vector()
        u = self.inputs.get_vector()

        if len(x) == 0:
            # system is memoryless and output depends only on input
            y = self.calc_outputs(u, time)
        else: 
            # pass the state into the output function
            # TODO: allow outputs that are a function of state and feedthrough input 
            # (or, rather, reconstruct system to seperate the feedforward component, assuming x & u are coupled)
            y = self.calc_outputs(x, time)

        self.outputs.set_values(y)

    def update_dynamics(self, time):
        x = self.states.get_vector()
        u = self.inputs.get_vector()
        return self.dynamics(x, u, time)
        
    def calc_outputs(self, state_or_input, time):
        pass

    def dynamics(self, x, u, time):
        pass

class SimulationEngine:
    def __init__(self):
        self.subsystem_list = [] # list of dictionaries 

    def add_subsystem(self, subsystem):
        self.subsystem_list.append({'subsystem': subsystem, 'connections': []})

    def connect(self, output_tuple, input_tuple):
        # output signal -> input signal
        # each argument tuple = (subsystem object, string or int index to identify the input or output)

        output_sys = output_tuple[0]
        output_signal_identifier = output_tuple[1]
        output_signal = output_sys.outputs[output_signal_identifier]

        input_sys = input_tuple[0]
        input_signal_identifier = input_tuple[1]
        input_signal = input_sys.inputs[input_signal_identifier]

        # check to see if subsystem already has connections
        for item in self.subsystem_list:
            # TODO: implement more efficient search through the list of dictionaries
            if item['subsystem'] == input_sys:
                item['connections'].append((output_signal, input_signal))
                break
        
    def update_outputs(self, time):
        # calls subsystem update_outputs functions while enforcing causality by rearranging the subsystem list, if necessary
        index = 0

        output_computed_list = [] # list of subsystems that have had their outputs computed, so that the count doesn't increase by more than one

        while index < len(self.subsystem_list):
            item = self.subsystem_list[index]

            # See if inputs can be assigned or if they already have been
            all_inputs_computed = True
            
            for connection in item['connections']:
                # check that all feeding outputs have been updated
                output_signal = connection[0]
                input_signal = connection[1]
                if output_signal.count != input_signal.count + 1:
                    all_inputs_computed = False
                    break
                
            if all_inputs_computed:
                for connection in item['connections']:
                    output_signal = connection[0]
                    input_signal = connection[1]
                    input_signal.value = output_signal.value

            # Compute outputs:
            if all_inputs_computed or len(item['connections']) == 0 or len(item['subsystem'].states) != 0:
                if item not in output_computed_list:
                    item['subsystem'].update_outputs(time)
                    output_computed_list.append(item)

            # Rearrange the list if necessary, or advance index       
            if all_inputs_computed == False:
                # Move the item at the current index to the end of the subsystem_list
                current_item = self.subsystem_list[index]
                self.subsystem_list.append(current_item)
                self.subsystem_list.pop(index)
            else:
                index += 1
                
    def populate_states(self, states):
        # unpack the state vector into the subsystems
        counter = 0
        for item in self.subsystem_list:
            if len(item['subsystem'].states) > 0:
                item['subsystem'].states.set_values(states[counter : counter + len(item['subsystem'].states)])
                counter += len(item['subsystem'].states)

    def dynamics(self, states, time):
        derivative = np.array([])

        # populate states
        self.populate_states(states)

        # compute outputs
        self.update_outputs(time)

        # compute dynamics
        for item in self.subsystem_list:
            xdot = item['subsystem'].update_dynamics(time)
            if xdot is not None:
                derivative = np.append(derivative, xdot)

        return derivative

    def reset_counts(self):
        for item in self.subsystem_list:
            for signal in item['subsystem'].states.signals:
                signal.count = 0
            for signal in item['subsystem'].inputs.signals:
                signal.count = 0
            for signal in item['subsystem'].outputs.signals:
                signal.count = 0

    def simulate(self, initial_states, time, dt=0.01):
        self.reset_counts()
        t0 = time[0] # tuple
        tf = time[1]
        t = np.arange(t0, tf, dt)
        state_trajectories = odeint(self.dynamics, initial_states, t)
        return state_trajectories



# left off - label plots and states