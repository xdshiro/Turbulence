import numpy as np
import matplotlib.pyplot as plt
import aotools
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation


def laguerre_gaussian(r, theta, z, l, p, w0, k):
    """
    Compute the electric field of an LG beam.

    Parameters:
    - r, theta: Radial and azimuthal coordinates
    - z: Propagation distance
    - l, p: LG beam indices
    - w0: Beam waist
    - k: Wavenumber

    Returns:
    - Electric field of the LG beam
    """
    # Beam width as a function of z
    w = w0 * np.sqrt(1 + (z * wavelength / (np.pi * w0 ** 2)) ** 2)
    
    # Radius of curvature
    if z == 0:
        R = np.inf
    else:
        R = z * (1 + (np.pi * w0 ** 2 / (z * wavelength)) ** 2)
    
    # Gouy phase shift
    zeta = np.arctan(z * wavelength / (np.pi * w0 ** 2))
    
    # Compute the field
    L = laguerre(p)
    amplitude = (np.sqrt(2) * r / w) ** l * np.exp(-r ** 2 / w ** 2) * L(2 * r ** 2 / w ** 2)
    phase = l * theta - k * r ** 2 / (2 * R) + (2 * p + np.abs(l) + 1) * zeta
    
    return amplitude * np.exp(1j * phase)

def gaussian_beam(x, y, w0, k, R, A=1):
    """Generate a Gaussian beam"""
    return A * np.exp(-(x**2 + y**2) / w0**2) #* np.exp(1j * k * (x**2 + y**2) / (2*R))


def free_space_propagation(E, dx, dy, wavelength, dz):
    """
    Propagate the electric field E in free space over a distance dz.

    Parameters:
    - E: Input electric field
    - dx, dy: Pixel spacing in x and y directions
    - wavelength: Wavelength of light
    - dz: Propagation distance

    Returns:
    - Electric field after propagation
    """
    nx, ny = E.shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    screen = phasescreen.ft_sh_phase_screen(r0 * 5e0, n_pixels, pxl_scale, L0, 0.01)
    im = plt.imshow(screen, extent=[-D / 2, D / 2, -D / 2, D / 2], cmap='jet')
    plt.title('ft_sh_phase_screen')
    plt.colorbar(im, label='', fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.show()
    # E_current *= np.exp(1j * screen)
    # Transfer function for free space propagation
    H = np.exp(-1j * np.pi * wavelength * dz * (FX ** 2 + FY ** 2))
    
    # Propagation in Fourier domain
    E_prop = np.fft.ifft2(np.fft.fft2(E) * (H + 1j * screen))
    
    return E_prop

def plot_field(E, extent=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    im0 = ax[0].imshow(np.abs(E) ** 2, extent=extent, cmap='magma')
    ax[0].set_title('|Amplitude|')
    fig.colorbar(im0, ax=ax[0], label='Amplitude', fraction=0.04, pad=0.02)
    
    im1 = ax[1].imshow(np.angle(E), extent=[-D / 2, D / 2, -D / 2, D / 2], cmap='jet')
    ax[1].set_title('Phase')
    fig.colorbar(im1, ax=ax[1], label='Phase (radians)', fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.show()
# Define simulation parameters
D = 100e-6 * 4
r0 = 0.1
L0 = 25.0 * 10
l0 = 1e-4
pxl_scale = 1e-6
n_pixels = int(D / pxl_scale)
wavelength = 650e-9  # Red light, for instance
k = 2 * np.pi / wavelength
w0 = 20e-6  # Beam waist in meters
R = 100  # Arbitrary radius of curvature

# Generate the phase screen
"""
r0 (float): r0 parameter of scrn in metres
N (int): Size of phase scrn in pxls
delta (float): size in Metres of each pxl
L0 (float): Size of outer-scale in metres
l0 (float): inner scale in metres
"""
screen = phasescreen.ft_sh_phase_screen(r0=5e-5, N=n_pixels, delta=pxl_scale,
                                        L0=L0 / 100, l0=1e-4)


# Generate coordinate system
x = np.linspace(-D/2, D/2, n_pixels)
y = np.linspace(-D/2, D/2, n_pixels)
X, Y = np.meshgrid(x, y)

# Generate Gaussian beam
E = gaussian_beam(X, Y, w0, k, R)

x = np.linspace(-D/2, D/2, n_pixels)
y = np.linspace(-D/2, D/2, n_pixels)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
l = 1
p = 0
z = 0  # Propagation distance
E = laguerre_gaussian(R, Theta, z, l, p, w0, k)
plot_field(E, np.array([-D / 2, D / 2, -D / 2, D / 2]) * 1e6)
pxl_mult = 1
L_pr = 0.01 / 6
screen = phasescreen.ft_sh_phase_screen(r0=5e-5, N=n_pixels, delta=pxl_scale,
                                        L0=L0, l0=l0)
E = E * np.exp(1j * screen)
E_current = opticalpropagation.angularSpectrum(E, wavelength, pxl_scale, pxl_scale * pxl_mult, L_pr)
plot_field(E_current, np.array([-pxl_mult * D / 2, pxl_mult * D / 2, -pxl_mult * D / 2, pxl_mult * D / 2]) * 1e6)
screen = phasescreen.ft_sh_phase_screen(r0=5e-5, N=n_pixels, delta=pxl_scale,
                                        L0=L0, l0=l0)
E = E_current * np.exp(1j * screen)

pxl_mult *= pxl_mult
E_current = opticalpropagation.angularSpectrum(E, wavelength, pxl_scale, pxl_scale * pxl_mult, L_pr)
plot_field(E_current, np.array([-pxl_mult * D / 2, pxl_mult * D / 2, -pxl_mult * D / 2, pxl_mult * D / 2]) * 1e6)
screen = phasescreen.ft_sh_phase_screen(r0=5e-5, N=n_pixels, delta=pxl_scale,
                                        L0=L0, l0=l0)
E = E_current * np.exp(1j * screen)
pxl_mult *= pxl_mult
E_current = opticalpropagation.angularSpectrum(E, wavelength, pxl_scale, pxl_scale * pxl_mult, L_pr)
plot_field(E_current, np.array([-pxl_mult * D / 2, pxl_mult * D / 2, -pxl_mult * D / 2, pxl_mult * D / 2]) * 1e6)
screen = phasescreen.ft_sh_phase_screen(r0=5e-5, N=n_pixels, delta=pxl_scale,
                                        L0=L0, l0=l0)
E = E_current * np.exp(1j * screen)
pxl_mult *= pxl_mult
E_current = opticalpropagation.angularSpectrum(E, wavelength, pxl_scale, pxl_scale * pxl_mult, L_pr)
plot_field(E_current, np.array([-pxl_mult * D / 2, pxl_mult * D / 2, -pxl_mult * D / 2, pxl_mult * D / 2]) * 1e6)


exit()
# Parameters
wavelength = 650e-9
k = 2 * np.pi / wavelength
w0 = 0.5  # Beam waist
l = 1
p = 0
D = 5 * w0  # Domain size for visualization
n_pixels = 400
z = 0  # Propagation distance

# Generate coordinate system
x = np.linspace(-D/2, D/2, n_pixels)
y = np.linspace(-D/2, D/2, n_pixels)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
E_lg = laguerre_gaussian(R, Theta, z, l, p, w0, k)
# Propagate through the phase screen
E_prime = E * np.exp(1j * screen)


# Define necessary parameters again
pxl_scale = 0.01  # Pixel scale in meters (i.e., resolution of the phase screen)

# Simulation parameters
total_distance = .10  # Total propagation distance in meters
n_steps = 1  # Number of propagation steps
dz = total_distance / n_steps  # Step size

E_current = E_lg.copy()  # Start with the initial LG beam

# Simulate propagation through turbulence
for _ in range(n_steps):
    # Free-space propagation over dz
    E_current = free_space_propagation(E_current, pxl_scale, pxl_scale, wavelength, dz)
    
    # Apply the phase screen (using the mock phase screen for demonstration)
    
    
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(np.abs(E_current)**2, extent=[-D/2, D/2, -D/2, D/2], cmap='gray')
ax[0].set_title('Intensity After Turbulence')
fig.colorbar(im0, ax=ax[0], label='Intensity', fraction=0.04, pad=0.02)

im1 = ax[1].imshow(np.angle(E_current), extent=[-D/2, D/2, -D/2, D/2], cmap='jet')
ax[1].set_title('Phase After Turbulence')
fig.colorbar(im1, ax=ax[1], label='Phase (radians)', fraction=0.04, pad=0.02)
plt.tight_layout()
plt.show()