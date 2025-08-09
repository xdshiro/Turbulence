import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

def create_waveguide_permittivity(nx, ny, dx, dy, core_width, core_height, eps_core, eps_clad):
    eps = np.ones((ny, nx)) * eps_clad
    cx, cy = nx // 2, ny // 2
    hw, hh = int(core_width / (2 * dx)), int(core_height / (2 * dy))
    eps[cy - hh:cy + hh, cx - hw:cx + hw] = eps_core
    return eps

def build_helmholtz_matrix(eps_r, omega, dx, dy):
    ny, nx = eps_r.shape
    N = nx * ny
    dx2 = dx**2
    dy2 = dy**2
    diag_main = np.zeros(N)
    diag_x1 = np.ones(N - 1)
    diag_x2 = np.ones(N - 1)
    diag_y1 = np.ones(N - nx)
    diag_y2 = np.ones(N - nx)

    eps_flat = eps_r.flatten()
    k0_sq_eps = (omega / c)**2 * eps_flat
    diag_main[:] = -2 / dx2 - 2 / dy2 + k0_sq_eps

    for i in range(1, nx):
        diag_x1[i - 1 + i * ny] = 1 / dx2
        diag_x2[i - 1 + i * ny] = 1 / dx2

    for i in range(ny, N):
        diag_y1[i - ny] = 1 / dy2
        diag_y2[i - ny] = 1 / dy2

    diagonals = [diag_main, diag_x1, diag_x2, diag_y1, diag_y2]
    offsets = [0, -1, 1, -nx, nx]
    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    return A

def solve_modes(eps_r, wavelength, dx, dy, num_modes=5):
    omega = 2 * np.pi * c / wavelength
    A = build_helmholtz_matrix(eps_r, omega, dx, dy)
    vals, vecs = eigs(A, k=num_modes, which='LR')
    neffs = np.sqrt(np.real(vals)) * wavelength / (2 * np.pi)
    modes = [vecs[:, i].reshape(eps_r.shape) for i in range(num_modes)]
    return neffs, modes

def plot_mode(mode, dx, dy, title='Mode', cmap='RdBu'):
    plt.figure()
    plt.imshow(np.real(mode), cmap=cmap, origin='lower', extent=[-mode.shape[1]*dx/2, mode.shape[1]*dx/2,
                                                                  -mode.shape[0]*dy/2, mode.shape[0]*dy/2])
    plt.colorbar(label='Re(E)')
    plt.title(title)
    plt.xlabel('x (µm)')
    plt.ylabel('y (µm)')
    plt.tight_layout()
    plt.show()

# --- Parameters ---
wavelengths = [1.31e-6, 1.55e-6]  # in meters
nx, ny = 200, 200
dx, dy = 0.02e-6, 0.02e-6  # grid resolution
core_width, core_height = 0.45e-6, 0.22e-6
eps_core = 3.48**2
eps_clad = 1.44**2

# --- Compute TE0, TM0 ---
for wl in wavelengths:
    eps = create_waveguide_permittivity(nx, ny, dx, dy, core_width, core_height, eps_core, eps_clad)
    neffs, modes = solve_modes(eps, wl, dx, dy, num_modes=5)
    print(f"Wavelength: {wl*1e9:.0f} nm, neffs: {np.sort(np.real(neffs))[::-1]}")

    plot_mode(modes[0], dx, dy, title=f"TE0 mode at {wl*1e9:.0f} nm")
    plot_mode(modes[1], dx, dy, title=f"TM0 mode at {wl*1e9:.0f} nm")

# --- Dispersion plot: Effective Index vs Wavelength ---
wl_range = np.linspace(1.3e-6, 1.6e-6, 30)
neffs_dispersion = []

for wl in wl_range:
    eps = create_waveguide_permittivity(nx, ny, dx, dy, core_width, core_height, eps_core, eps_clad)
    neffs, _ = solve_modes(eps, wl, dx, dy, num_modes=5)
    neffs_dispersion.append(np.sort(np.real(neffs))[::-1])

neffs_dispersion = np.array(neffs_dispersion)
plt.figure()
for i in range(5):
    plt.plot(wl_range * 1e9, neffs_dispersion[:, i], label=f"Mode {i}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Effective index")
plt.title("Effective indices of first 5 modes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()