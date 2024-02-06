import control as ct
import numpy as np
import matplotlib.pyplot as plt

# Define the system
A = [[0, 1], [-.2, -1]]
B = [[0], [1]]
C = [[1, 0]]
D = [[0]]
plant = ct.ss(A, B, C, D)
actuator = ct.tf(1, [.5, 1])
gain = 2.5
input = ct.tf(1, [1, 0])

sys = input*gain*actuator*plant

# Define the input
t = np.linspace(0, 5, 100)

# Simulate the system response
resp = ct.step_response(sys, T=t)
y = resp[1]

# Plot the response
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response')
plt.grid(True)
plt.show()