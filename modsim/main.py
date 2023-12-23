import numpy as np
from scipy.integrate import odeint

class Subsystem:
    def __init__(self, parameters={}):
        self.dim = 0
        self.states = np.array([])
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.parameters = parameters

    def update_outputs(self):
        pass

    def dynamics(self, time):
        pass

class Plant(Subsystem):

    def __init__(self):
        super().__init__()
        self.dim = 1

    def update_outputs(self):
        outputs = self.states
        self.outputs = outputs

    def dynamics(self, time):
        return self.parameters['a']*self.states + self.parameters['b']*self.inputs


class Actuator(Subsystem):

    def __init__(self):
        super().__init__()
        self.dim = 1

    def update_outputs(self):
        outputs = self.states
        self.outputs = outputs

    def dynamics(self, time):
        return 1/self.parameters['time constant'] * (self.inputs - self.states)

class StepInput(Subsystem):

    def update_outputs(self):
        self.outputs = 1.0

    def dynamics(self, time):
        pass

class SimulationEngine:
    def __init__(self):
        self.subsystems = []
        self.connections = []

    def add_subsystem(self, subsystem):
        self.subsystems.append(subsystem)

    def connect(self, input_subsystem, output_subsystem):
        self.connections.append((input_subsystem, output_subsystem))

    def populate_states(self, states):
        # unpack the state vector into the subsystems
        counter = 0
        for subsystem in self.subsystems:
            if subsystem.dim > 0:
                subsystem.states = states[counter]
                counter += 1
        return states

    def compute_outputs(self):
        for subsystem in self.subsystems:
            subsystem.update_outputs()

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