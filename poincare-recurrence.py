import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lorenz system dynamics
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions and time span
initial_state = [0.1, 0.0, 0.0]  # Initial state [x, y, z]
t_span = (0, 100)  # Time span for simulation

# Solve the Lorenz system using scipy's solve_ivp
sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)

# Time array for plotting
t_plot = np.linspace(t_span[0], t_span[1], 10000)

# Evaluate the solution at the plotting time points
states_plot = sol.sol(t_plot)

# Plot the phase space trajectory (x vs. z)
plt.figure(figsize=(8, 6))
plt.plot(states_plot[0], states_plot[2], label='Lorenz System Trajectory', color='blue')
plt.scatter(states_plot[0][0], states_plot[2][0], color='red', label='Initial State')  # Mark initial state
plt.xlabel('x')
plt.ylabel('z')
plt.title('Lorenz System Phase Space (x vs. z)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the time evolution of x, y, and z
plt.figure(figsize=(12, 6))
plt.plot(t_plot, states_plot[0], label='x', color='blue')
plt.plot(t_plot, states_plot[1], label='y', color='green')
plt.plot(t_plot, states_plot[2], label='z', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Lorenz System Time Evolution')
plt.legend()
plt.grid(True)
plt.show()
