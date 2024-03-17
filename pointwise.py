import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 1000  # Number of steps in the random walk
initial_position = 0  # Initial position of the walker
prob_right = 0.5  # Probability of moving right (otherwise, move left)

# Simulate the random walk
trajectory = np.zeros(num_steps + 1)
trajectory[0] = initial_position
for t in range(num_steps):
    if np.random.rand() < prob_right:
        trajectory[t + 1] = trajectory[t] + 1
    else:
        trajectory[t + 1] = trajectory[t] - 1

# Compute the time average along the trajectory
time_average = np.cumsum(trajectory) / np.arange(1, num_steps + 2)

# Plot the trajectory and time average
plt.figure(figsize=(10, 5))
plt.plot(np.arange(num_steps + 1), trajectory, label='Trajectory')
plt.plot(np.arange(num_steps + 1), time_average, label='Time Average', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.title('One-Dimensional Random Walk and Time Average')
plt.grid(True)
plt.show()
