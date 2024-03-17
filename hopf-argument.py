import numpy as np
import matplotlib.pyplot as plt

# Define the periodic function
def periodic_function(x, period):
    return np.sin(2 * np.pi * x / period)  # Example: sin function with given period

# Compute the Fourier coefficients of the periodic function
def compute_fourier_coefficients(f_values, period, num_coefficients):
    coefficients = []
    for n in range(-num_coefficients, num_coefficients + 1):
        coeff = np.sum(f_values * np.exp(-1j * 2 * np.pi * n * np.arange(len(f_values)) / len(f_values)))
        coefficients.append(coeff / len(f_values))
    return np.array(coefficients)

# Compute the Fourier transform of the periodic function
def compute_fourier_transform(coefficients, period):
    freqs = np.fft.fftfreq(len(coefficients), d=period / len(coefficients))
    ft_values = np.fft.fftshift(np.fft.fft(coefficients))
    return freqs, ft_values

# Parameters
period = 2 * np.pi  # Period of the periodic function
num_points = 1000  # Number of points for plotting and FFT
num_coefficients = 10  # Number of Fourier coefficients to compute

# Compute the periodic function and its Fourier coefficients
x = np.linspace(0, period, num_points, endpoint=False)
f_values = periodic_function(x, period)
fourier_coeffs = compute_fourier_coefficients(f_values, period, num_coefficients)

# Compute the Fourier transform of the Fourier coefficients
freqs, ft_values = compute_fourier_transform(fourier_coeffs, period)

# Plot the periodic function and its Fourier transform
plt.figure(figsize=(12, 6))

# Plot the periodic function
plt.subplot(2, 1, 1)
plt.plot(x, f_values, label='Periodic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Periodic Function')
plt.grid(True)
plt.legend()

# Plot the Fourier transform
plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(ft_values), label='|FT(f)|')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform')
plt.grid(True)
plt.legend()

# Highlight the zeros of the Fourier transform
zero_indices = np.where(np.abs(ft_values) < 1e-10)[0]
plt.scatter(freqs[zero_indices], np.abs(ft_values[zero_indices]), color='red', label='Zeros')
plt.legend()

plt.tight_layout()
plt.show()
