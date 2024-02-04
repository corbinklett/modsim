import modsim as ms
import numpy as np

# Use the modsim package to create a simple open loop system

# Example usage
time_constant = 0.5
actuator = ms.Actuator(time_constant)

k_spring = 1
k_damp = .2
A = np.array([[0, 1], [-k_damp, -k_spring]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
plant = ms.LinearSystem(A, B, C)
step = ms.StepInput()
gain = ms.Gain(2.5)

sim = ms.SimulationEngine()
sim.add_subsystem(actuator)
sim.add_subsystem(gain)
sim.add_subsystem(step)
sim.add_subsystem(plant, [0, 1]) # TODO: change to x0 keyword argument

gain.outputs.names_units("y")
actuator.inputs.names_units("u", "meters")
actuator.outputs.names_units("y", "Newtons")
plant.inputs.names_units("u", "Newtons")
plant.states.names_units(["x1", "x2"], ["meters", "m/s"])

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