import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelength = 0.0000006328  # Wavelength of the beam in meters (e.g., 632.8 nm for red light)
k = 2 * np.pi / wavelength  # Wavenumber
w0 = 0.001  # Beam waist at z=0 in meters
z = 50  # Propagation distance in meters

# Spatial grid
x = np.linspace(-0.2, 0.2, 2500)
y = np.linspace(-0.2, 0.2, 2500)
X, Y = np.meshgrid(x, y)
r_squared = X**2 + Y**2

# Gaussian beam at z=0 (LG_00 mode)
def gaussian_beam_z0(X, Y, w0):
    return np.exp(-r_squared / w0**2)

# Gaussian beam after propagation at distance z
def gaussian_beam_z(X, Y, w0, z, wavelength):
    zr = np.pi * w0**2 / wavelength  # Rayleigh range
    wz = w0 * np.sqrt(1 + (z / zr)**2)  # Beam waist at distance z
    Rz = z * (1 + (zr / z)**2) if z != 0 else np.inf  # Radius of curvature
    phi_z = np.arctan(z / zr)  # Gouy phase
    return (w0 / wz) * np.exp(-r_squared / wz**2) * np.exp(1j * (k * r_squared / (2 * Rz) - k * z + phi_z))

# 2D Fourier transform of the beam to get the spectrum
def compute_spectrum(beam):
    spectrum = np.fft.fftshift(np.fft.fft2(beam))
    return np.abs(spectrum)

# Beam at z=0
beam_z0 = gaussian_beam_z0(X, Y, w0)
spectrum_z0 = compute_spectrum(beam_z0)

# Beam at z=some distance
beam_z = gaussian_beam_z(X, Y, w0, z, wavelength)
spectrum_z = compute_spectrum(beam_z)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot Gaussian beam at z=0
im0 = axs[0, 0].imshow(np.abs(beam_z0), extent=[x.min(), x.max(), y.min(), y.max()])
axs[0, 0].set_title("Gaussian Beam at z=0")
axs[0, 0].set_xlabel("x (m)")
axs[0, 0].set_ylabel("y (m)")
plt.colorbar(im0, ax=axs[0, 0])

# Plot spectrum at z=0
im1 = axs[0, 1].imshow(spectrum_z0, extent=[x.min(), x.max(), y.min(), y.max()])
axs[0, 1].set_title("2D Spectrum at z=0")
axs[0, 1].set_xlabel("kx (1/m)")
axs[0, 1].set_ylabel("ky (1/m)")
plt.colorbar(im1, ax=axs[0, 1])

# Plot Gaussian beam at z=some distance
im2 = axs[1, 0].imshow(np.abs(beam_z), extent=[x.min(), x.max(), y.min(), y.max()])
axs[1, 0].set_title(f"Gaussian Beam at z={z} m")
axs[1, 0].set_xlabel("x (m)")
axs[1, 0].set_ylabel("y (m)")
plt.colorbar(im2, ax=axs[1, 0])

# Plot spectrum at z=some distance
im3 = axs[1, 1].imshow(spectrum_z, extent=[x.min(), x.max(), y.min(), y.max()])
axs[1, 1].set_title(f"2D Spectrum at z={z} m")
axs[1, 1].set_xlabel("kx (1/m)")
axs[1, 1].set_ylabel("ky (1/m)")
plt.colorbar(im3, ax=axs[1, 1])

plt.tight_layout()
plt.show()