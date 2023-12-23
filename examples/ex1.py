import modsim as ms
import numpy as np
import matplotlib as plt

# Example usage
actuator = ms.Actuator()

actuator.parameters = {'time constant': 1}
plant = ms.Plant()
plant.parameters = {'a': -0.3, 'b': 2}
step = ms.StepInput()

sim = ms.SimulationEngine()
sim.add_subsystem(actuator)
sim.add_subsystem(plant)
sim.add_subsystem(step)

sim.connect(actuator, plant) # input -> output
sim.connect(step, actuator) # input -> output

x0 = np.array([0,0])
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