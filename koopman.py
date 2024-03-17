import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define the nonlinear dynamical system (example: a nonlinear oscillator)
def nonlinear_system(x, t):
    return np.array([-x[0] + x[1]**2, -x[1] - x[0]**2])

# Generate data for the nonlinear system
t_span = np.linspace(0, 10, 1000)  # Time span
x_init = np.array([1.0, 1.0])  # Initial condition
data = np.zeros((len(t_span), len(x_init)))
data[0] = x_init
for i in range(1, len(t_span)):
    data[i] = data[i - 1] + nonlinear_system(data[i - 1], t_span[i - 1]) * (t_span[i] - t_span[i - 1])

# Define the basis functions for the EDMD
def basis_functions(x):
    return np.array([1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]])

# Construct the extended state matrix Phi
Phi = basis_functions(data[:-1].T)

# Compute the Koopman operator matrix K
K = np.dot(Phi.T, Phi)

# Compute the eigendecomposition of the Koopman operator
eigenvalues, eigenvectors = eigh(K)

# Plot the eigenvalues of the Koopman operator
plt.figure(figsize=(8, 6))
plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), label='Eigenvalues', color='blue')
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Eigenvalues of Koopman Operator')
plt.legend()
plt.grid(True)
plt.show()
