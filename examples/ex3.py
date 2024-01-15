import modsim as ms
import numpy as np
import matplotlib as plt

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
sim.add_subsystem(plant)
sim.add_subsystem(step)
sim.add_subsystem(gain)

gain.outputs.names_units("k", "V/m")
actuator.inputs.names_units("u", "V")

# draw individual lines between blocks
# vector lines too?
# TODO: update so this just needs to be gain['k'] which automatically grabs the output
# enforce that it only connects and output to an input
# output -> input
returns an object that contains both the subsystme and the signal
sim.connect(gain.outputs["k"], actuator.inputs["u"])
sim.connect(step.outputs[0], gain.inputs[0])
sim.connect(actuator.outputs[0], plant.inputs[0])
# next - do a closed loop system, and try multiple inputs/outputs

x0 = np.array([0,0,0])
res = sim.simulate(x0, (0, 10))

# left off - plot it. Then generalize dimension and add labels
# then enforce types?

import matplotlib.pyplot as plt

# Plotting the outputs
time = np.arange(0, 10, 0.01)  # Time array for plotting
output1 = res[:, 0]  # Extracting the first output
output2 = res[:, 1]  # Extracting the second output

plt.figure(figsize=(10, 6))
plt.plot(time, output1, label='Output 1')
plt.plot(time, output2, label='Output 2')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Simulation Outputs')
plt.legend()
plt.grid(True)
plt.show()