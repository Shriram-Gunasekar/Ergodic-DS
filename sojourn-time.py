import numpy as np

# Define the transition matrix for the Markov chain
transition_matrix = np.array([[0.8, 0.2, 0.0],
                              [0.1, 0.7, 0.2],
                              [0.3, 0.1, 0.6]])

# Define the initial state probabilities
initial_state = np.array([0.5, 0.3, 0.2])

# Define the state of interest (state 1 in this example)
state_of_interest = 1

# Define the number of simulations and maximum time steps
num_simulations = 1000
max_time_steps = 100

# Initialize an array to store the sojourn times
sojourn_times = np.zeros(num_simulations)

# Simulate the Markov chain and calculate sojourn times
for i in range(num_simulations):
    current_state = np.random.choice(len(initial_state), p=initial_state)
    time_steps = 0
    while time_steps < max_time_steps:
        next_state = np.random.choice(len(initial_state), p=transition_matrix[current_state])
        if next_state == state_of_interest:
            sojourn_times[i] = time_steps + 1  # Add 1 to account for the initial visit
            break
        current_state = next_state
        time_steps += 1

# Calculate the average sojourn time
average_sojourn_time = np.mean(sojourn_times)

print(f"Average Sojourn Time in State {state_of_interest}: {average_sojourn_time} time steps")
