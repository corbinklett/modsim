import modsim as ms
import numpy as np
import matplotlib.pyplot as plt

# Example usage
actuator = ms.Actuator(.5)
A = np.array([[0, 1], [-.2, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
plant = ms.LinearSystem(A, B, C)
step = ms.StepInput()
gain = ms.Gain(2.5)

sim = ms.SimulationEngine()
sim.add_subsystem(actuator)
sim.add_subsystem(gain)
sim.add_subsystem(step)
sim.add_subsystem(plant, [0, 1])

gain.outputs.names_units("y", "mA")
actuator.inputs.names_units("u", "V")

# output -> input
sim.connect(gain.outputs["y"], actuator.inputs["u"])
sim.connect(step.outputs[0], gain.inputs[0])
sim.connect(actuator.outputs[0], plant.inputs[0])

actuator.initial_states = [.1]  # TODO: make initial_states an attribute with setter that changes states

t0 = 0
tf = 10
sol = sim.simulate(t0, tf)
res = sol['ode_solution']
time = res.t

#sim.plot(gain.outputs['y'], actuator.outputs[0], plant.states[0], plant.states[1])
sim.plot()