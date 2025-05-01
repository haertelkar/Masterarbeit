# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Define parameters
wavelength = 0.025  # Electron wavelength in Å (for ~200 keV electrons)
defocus_values = [0, -50, -100]  # Defocus values in Å
convergence_angles = [10, 20, 30]  # Convergence semiangles in mrad
x = np.linspace(-10, 10, 500)  # Spatial coordinate in Å
X, Y = np.meshgrid(x, x)
r = np.sqrt(X**2 + Y**2)

# Improved simulation using Fourier optics approach

# Define grid
grid_size = 256  # Number of pixels
real_space_extent = 10  # Å (size of the real-space grid)
k_space_extent = 1 / (2 * (real_space_extent / grid_size))  # Reciprocal space extent

x = np.linspace(-real_space_extent, real_space_extent, grid_size)
X, Y = np.meshgrid(x, x)
r = np.sqrt(X**2 + Y**2)

# Function to compute the STEM probe intensity using Fourier optics
def compute_probe_intensity(defocus, conv_angle):
    k_max = (conv_angle / 1000) / wavelength  # Convert mrad to Å^-1
    kx = np.fft.fftfreq(grid_size, d=(2 * real_space_extent / grid_size)) * (2 * np.pi)
    ky = np.fft.fftfreq(grid_size, d=(2 * real_space_extent / grid_size)) * (2 * np.pi)
    KX, KY = np.meshgrid(kx, ky)
    k_r = np.sqrt(KX**2 + KY**2)
    
    # Aperture function (circular aperture in reciprocal space)
    aperture = k_r <= k_max
    
    # Phase shift due to defocus (quadratic phase term)
    phase_shift = np.exp(1j * np.pi * wavelength * defocus * k_r**2) * aperture
    
    # Inverse Fourier Transform to get real-space probe intensity
    probe_wave = np.fft.ifft2(phase_shift)
    intensity = np.abs(probe_wave)**2
    
    return np.fft.fftshift(intensity)  # Centered intensity distribution

# Generate probe intensities for different defocus and convergence angles
fig, axes = plt.subplots(len(defocus_values), len(convergence_angles), figsize=(12, 10))

for i, df in enumerate(defocus_values):
    for j, angle in enumerate(convergence_angles):
        intensity = compute_probe_intensity(df, angle)
        ax = axes[i, j]
        im = ax.imshow(intensity, extent=(-real_space_extent, real_space_extent, 
                                          -real_space_extent, real_space_extent), 
                       cmap='inferno', origin='lower')
        ax.set_title(f"Conv. Angle = {angle} mrad\nDefocus = {df} Å")
        ax.set_xlabel("Å")
        ax.set_ylabel("Å")

plt.tight_layout()
plt.colorbar(im, ax=axes.ravel().tolist(), label="Intensity")
plt.show()


