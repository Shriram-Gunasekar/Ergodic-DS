import numpy as np
import matplotlib.pyplot as plt

def compute_stationary_distribution(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_vector = np.abs(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    stationary_vector /= np.sum(stationary_vector)
    return stationary_vector

def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    states = np.zeros((num_steps + 1, len(initial_state)))
    states[0] = initial_state
    for t in range(num_steps):
        states[t + 1] = np.dot(states[t], transition_matrix)
    return states

# Define the transition matrix for a simple Markov chain
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])

# Initial state probabilities
initial_state = np.array([0.5, 0.5])

# Compute the stationary distribution
stationary_dist = compute_stationary_distribution(transition_matrix)
print("Stationary Distribution:", stationary_dist)

# Simulate the Markov chain over time
num_steps = 1000
states = simulate_markov_chain(transition_matrix, initial_state, num_steps)

# Plot the state trajectory
plt.figure(figsize=(10, 5))
plt.plot(states[:, 0], label='State 0')
plt.plot(states[:, 1], label='State 1')
plt.xlabel('Time Step')
plt.ylabel('Probability')
plt.title('Markov Chain Trajectory')
plt.grid(True)
plt.legend()
plt.show()
