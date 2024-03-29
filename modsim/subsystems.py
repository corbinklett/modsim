from .main import Subsystem
import numpy as np

class LinearSystem(Subsystem):
    def __init__(self, A, B, C, D=None, name='Linear System'):
        # TODO: enforce input requirements. Technically, any of the inputs could be None.
        parameters = {'A': A, 'B': B, 'C': C}
        if D is None:
            parameters['D'] = np.zeros((C.shape[0], B.shape[1]))
        super().__init__(A.shape[0], B.shape[1], C.shape[0], parameters, name=name)
        # optionally, name signals here

    def calc_outputs(self, x, time):
        outputs = self.parameters['C'] @ x
        # TODO: allow feedthrough terms
        # if self.parameters['D'] is not None:
        #     outputs += self.parameters['D'] @ u
        return outputs
        
    def dynamics(self, x, u, time):
        xdot = np.zeros(len(x))
        if self.parameters['A'] is not None:
            xdot += self.parameters['A'] @ x
        if self.parameters['B'] is not None:
            xdot += self.parameters['B'] @ u
        return xdot

class Actuator(Subsystem):

    def __init__(self, time_constant, name='Actuator'):
        parameters = {'time constant': time_constant}
        super().__init__(1, 1, 1, parameters, name=name)

    def calc_outputs(self, x, time):
        return x

    def dynamics(self, x, u, time):
        # TODO: add rate limits and saturation
        return 1/self.parameters['time constant'] * (u - x)

class StepInput(Subsystem):
    def __init__(self, name='Step Input'):
        super().__init__(0, 0, 1, name=name)

    def calc_outputs(self, u, time):
        return 1.0

    def dynamics(self, x, u, time):
        pass

class Gain(Subsystem):
    def __init__(self, k, name='Gain'):
        parameters = {'k': k}
        super().__init__(0, 1, 1, parameters, name=name)

    def calc_outputs(self, u, time):
        return self.parameters['k'] * u

    def dynamics(self, x, u, time):
        pass
