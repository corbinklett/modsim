#import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.integrate import solve_ivp


class Signal: 
    def __init__(self, subsystem=None):
        self._value: float = 0.0  # scalar
        self.name: str = ""
        self.unit: str = ""
        self.count: int = 0 # keeps track of assignments to this signal
        self.subsystem = subsystem # weird way to do it, but gives the signal knowledge of its owning subsystem
        #TODO: add description for ingestion by LLM

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self.count += 1
        

class SignalSet:
    def __init__(self, dimension, subsystem=None):
        self.signals: List[Signal] = [Signal(subsystem) for _ in range(dimension)]
        self.dimension = dimension
        self.subsystem = subsystem # weird way to do it, but gives the signal set knowledge of its owning subsystem

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

    def reset_signal_counts(self):
        for signal in self.signals:
            signal.count = 0
    
class Subsystem:
    # TODO: allow subsystems to contain subsystems
    def __init__(self, dim_states, dim_inputs, dim_outputs, parameters={}, name =''):
        self.dim_states = dim_states
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs

        self.states = SignalSet(dim_states, self) # defaults to 0.0
        self._initial_states = self.states.get_vector()
        
        self.inputs = SignalSet(dim_inputs, self)
        self.outputs = SignalSet(dim_outputs, self)
        self.parameters = parameters

        self.name = name

    def __getitem__(self, key):
        return self, key

    @property
    def initial_states(self):
        return self._initial_states

    @initial_states.setter
    def initial_states(self, vector):
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        self._initial_states = vector
        self.set_states(vector)
    
    def set_states(self, vector):        
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)   
        self.states.set_values(vector)

    def get_states(self):
        return self.states.get_vector()
        
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
        self.arranged_subsystem_list = []
        self.time_range = ()
        self.sim_solution = {}

    # TODO: implement __getitem__ method to return a subsystem object; may need to name subsystems first
    # def __getitem__(self, key):
    #     for item in self.subsystem_list:
    #         if item['subsystem'] == key:
    #             return item
    #     raise KeyError("Subsystem not found")
    
    def add_subsystem(self, subsystem, initial_states=None):
        # Check for duplicate subsystem names
        for item in self.subsystem_list:
            if item['subsystem'].name == subsystem.name:
                raise ValueError(f"Duplicate subsystem name: {subsystem.name}")

        self.subsystem_list.append({'subsystem': subsystem, 'connections': []})

        if initial_states is not None:
            subsystem.initial_states = initial_states

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

        while index < len(self.arranged_subsystem_list):
            item = self.arranged_subsystem_list[index]

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
                current_item = self.arranged_subsystem_list[index]
                self.arranged_subsystem_list.append(current_item)
                self.arranged_subsystem_list.pop(index)
            else:
                index += 1
                
    def populate_states(self, states):
        # unpack the entire state vector into the seperate subsystems
        counter = 0
        for item in self.subsystem_list:
            if item['subsystem'].dim_states > 0:
                item['subsystem'].set_states(states[counter : counter + item['subsystem'].dim_states])
                counter += item['subsystem'].dim_states

    def dynamics(self, time, states):
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

    def reset_signal_counts(self):
        for item in self.subsystem_list:
            for signal in item['subsystem'].states.signals:
                signal.count = 0
            for signal in item['subsystem'].inputs.signals:
                signal.count = 0
            for signal in item['subsystem'].outputs.signals:
                signal.count = 0

    def simulate(self, t0, tf):
        self.sim_solution = {}
        self.reset_signal_counts()
        self.arranged_subsystem_list = [item for item in self.subsystem_list] # does not copy objects; creates a new list of references to the same objects
        self.time_range = (t0, tf)

        # populate initial states
        initial_states = np.concatenate([item['subsystem'].get_states() for item in self.subsystem_list])

        ode_solution = solve_ivp(self.dynamics, self.time_range, initial_states)

        # save data for plotting
        self.package_solution(ode_solution)

        return self.sim_solution
    
    def package_solution(self, ode_solution):
        # TODO: document the sim_solution dictionary
        # Collect inputs and outputs from all subsystems. Format just like the solve_ivp solution

        self.sim_solution['ode_solution'] = ode_solution
        self.sim_solution['t'] = ode_solution.t
        self.sim_solution['trajectories'] = {}

        sol_length = len(ode_solution.t)

        for item in self.subsystem_list:
            subsystem = item['subsystem']   
            self.sim_solution['trajectories'][subsystem.name] = {}
            
            self.sim_solution['trajectories'][subsystem.name]['states'] = [] # list of dictionaries with state names, units, and values
            for signal in subsystem.states.signals:
                # TODO: add ability to get state by name as well as index
                self.sim_solution['trajectories'][subsystem.name]['states'].append({'name': signal.name, 'unit': signal.unit, 'values': np.empty(sol_length)})
            
            self.sim_solution['trajectories'][subsystem.name]['inputs'] = []
            for signal in subsystem.inputs.signals:
                self.sim_solution['trajectories'][subsystem.name]['inputs'].append({'name': signal.name, 'unit': signal.unit, 'values': np.empty(sol_length)})
            
            self.sim_solution['trajectories'][subsystem.name]['outputs'] = []
            for signal in subsystem.outputs.signals:
                self.sim_solution['trajectories'][subsystem.name]['outputs'].append({'name': signal.name, 'unit': signal.unit, 'values': np.empty(sol_length)})

        for time_index, time in enumerate(ode_solution.t):
            states = ode_solution.y[:, time_index]
            self.populate_states(states)
            self.update_outputs(time)

            for item in self.subsystem_list:
                subsystem = item['subsystem']
                for state_index in range(subsystem.dim_states):
                    self.sim_solution['trajectories'][subsystem.name]['states'][state_index]['values'][time_index] = subsystem.states[state_index].value
                for input_index in range(subsystem.dim_inputs):
                    self.sim_solution['trajectories'][subsystem.name]['inputs'][input_index]['values'][time_index] = subsystem.inputs[input_index].value
                for output_index in range(subsystem.dim_outputs):
                    self.sim_solution['trajectories'][subsystem.name]['outputs'][output_index]['values'][time_index] = subsystem.outputs[output_index].value
                 
        return self.sim_solution
    
    def plot(self, plot_list=None, time_range=None): how to give this an artibrary number of args?
           sim.plot(actuator.states[0], actuator.outputs['y']) # plots against time

        if plot_list is None:
            plot_list = self.subsystem_list

        if time_range is None:
            time_range = self.time_range

        # recompute outputs and inputs
        self.reset_signal_counts()


        for item in plot_list:
            subsystem = item['subsystem']
            states = subsystem.states
            inputs = subsystem.inputs
            outputs = subsystem.outputs

            # loop through the state trajectory solution
            

            # Create a new figure and subplots for each subsystem
            fig, axs = plt.subplots(3, 1, figsize=(8, 6))
            fig.suptitle(f"Subsystem: {subsystem.__class__.__name__}")

            # Plot states
            axs[0].plot(time_range, states.get_values(), label="States")
            axs[0].set_ylabel("State Values")
            axs[0].legend()

            # Plot inputs
            axs[1].plot(time_range, inputs.get_values(), label="Inputs")
            axs[1].set_ylabel("Input Values")
            axs[1].legend()

            # Plot outputs
            axs[2].plot(time_range, outputs.get_values(), label="Outputs")
            axs[2].set_xlabel("Time")
            axs[2].set_ylabel("Output Values")
            axs[2].legend()

            # Show the plot
            plt.show()
        