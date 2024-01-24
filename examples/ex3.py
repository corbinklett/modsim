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

gain.outputs.names_units("y")
actuator.inputs.names_units("u", "V")

# output -> input
sim.connect(gain["y"], actuator["u"])
sim.connect(step[0], gain[0])
sim.connect(actuator[0], plant[0])

actuator.initial_states = [.1]  # TODO: make initial_states an attribute with setter that changes states

t0 = 0
tf = 10
sol = sim.simulate(t0, tf)
res = sol['ode_solution']
time = res.t

# import matplotlib.pyplot as plt

# Plotting the outputs
output1 = res.y[0]  # Extracting the first output
output2 = res.y[1]  # Extracting the second output
output3 = res.y[2]  # Extracting the third output

# sim.plot()

plt.figure(figsize=(10, 6))
plt.plot(time, output1, label='Output 1')
plt.plot(time, output2, label='Output 2')
plt.plot(time, output3, label='Output 3')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Simulation Outputs')
plt.legend()
plt.grid(True)
plt.show()

# # next - do a closed loop system, and try multiple inputs/outputs
# # name the states as well?
# save outputs to csv