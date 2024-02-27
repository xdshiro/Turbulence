import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
file_names = [
    'x525_y735',
    'x525_y550',
    'x345_y745',
    'x300_y525'
]
for file_name in file_names:
    file_path_imag = f'Delete_Jiannan\\{file_name}_imagEy.csv'
    file_path_real = f'Delete_Jiannan\\{file_name}_realEy.csv'

    # Reading CSV file, assuming the data starts directly without headers
    # Adjust 'skiprows' as needed if there are additional header lines
    data_real = pd.read_csv(file_path_real, skiprows=8, header=None)
    data_imag = pd.read_csv(file_path_imag, skiprows=8, header=None)
    # Extracting X, Y, and the fourth column (field amplitude)
    x_real = data_real[0].values
    y_real = data_real[1].values
    field_real = data_real[3].values  # Adjust the column index if necessary

    x_imag = data_imag[0].values
    y_imag = data_imag[1].values
    field_imag = data_imag[3].values  # Adjust the column index if necessary
    # Creating a 2D grid for interpolation
    xi = np.linspace(x_imag.min() * 0.99, x_imag.max() * 0.99, 500)
    yi = np.linspace(y_imag.min() * 0.99, y_imag.max() * 0.99, 500)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolating the field onto the grid
    E_real = griddata((x_real, y_real), field_real, (xi, yi), method='linear')
    E_imag = griddata((x_imag, y_imag), field_imag, (xi, yi), method='linear')
    E = E_real + 1j * E_imag
    E = E / np.max(np.abs(E))
    amplitude = np.abs(E)  # Amplitude of the complex field
    phase = np.angle(E)  # Phase of the complex field
    weights = amplitude  # Using amplitude as weights
    weights = amplitude ** 2
    weighted_sum_of_phases = np.sum(weights * phase)
    total_weights = np.sum(weights)
    weighted_avg_phase = weighted_sum_of_phases / total_weights
    # print("Weighted average phase (int):", weighted_avg_phase / np.pi)
    middle_indices = np.array(E.shape) // 2
    middle_phase = phase[middle_indices[0], middle_indices[1]]

    # Find the point of maximum amplitude
    max_amp_index = np.unravel_index(np.argmax(amplitude, axis=None), amplitude.shape)
    max_amp_phase = phase[max_amp_index]

    # print("Phase at the middle of the field:", middle_phase / np.pi)
    print("Phase at the point of maximum amplitude:", max_amp_phase / np.pi)
    desired_size = 3500  # Choose based on desired frequency resolution
    Z = np.fft.fft2(E, s=(desired_size, desired_size))  # Zero-padding to increase N
    Z_shifted = np.fft.fftshift(Z)  # Shift zero frequency components to the center

    # Frequency axes
    freq_x = np.fft.fftshift(np.fft.fftfreq(desired_size, d=(xi[0, 1] - xi[0, 0])))
    freq_y = np.fft.fftshift(np.fft.fftfreq(desired_size, d=(yi[1, 0] - yi[0, 0])))

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(np.abs(Z_shifted)), extent=(freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()))
    plt.colorbar()
    plt.title('Magnitude Spectrum with Zero Padding')
    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')
    plt.show()

    break
    # Assuming FFT has been performed and Z_shifted obtained
    from scipy.ndimage import zoom
    pad = 2000
    z_padded = np.pad(E, pad_width=((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    z_rescaled = zoom(z_padded, 0.05)

    # Now perform FFT on the rescaled field
    Z_shifted_rescaled = np.fft.fftshift(np.fft.fft2(z_rescaled))
    magnitude_log = (np.abs(Z_shifted_rescaled))

    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude_log, cmap='gray')
    plt.colorbar()
    plt.title("Log-scaled Magnitude Spectrum")
    plt.show()
    continue
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, np.abs(E), levels=50, cmap='viridis')
    plt.colorbar(label='Field Amplitude')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Interpolated 2D Field Amplitude')
    plt.tight_layout()

    plt.show()
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, np.angle(E), levels=50, cmap='jet')
    plt.colorbar(label='Field Phase')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Interpolated 2D Field Phase')
    plt.tight_layout()
    plt.show()


    def direct_dft2(signal):
        M, N = signal.shape
        dft2d = np.zeros((M, N), dtype=complex)
        m = np.arange(M).reshape((M, 1))
        n = np.arange(N).reshape((N, 1))
        for u in range(M):
            for v in range(N):
                dft2d[u, v] = np.sum(signal * np.exp(- 2j * np.pi * ((u * m / M) + (v * n / N))))
        return dft2d


    # E_padded = np.pad(E, ((-2 * len(E), 2 * len(E)), (-2 * len(E[0]), 2 * len(E[0]))), 'constant')
    # plt.imshow(np.abs(E_padded))
    # plt.show()
    Z = np.fft.fft2(E)
    Z_shifted = np.fft.fftshift(Z)
    # Z_shifted = direct_dft2(E)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(Z_shifted))
    plt.colorbar()
    plt.title("Magnitude of the 2D Fourier Transform")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.tight_layout()

    plt.show()
    plt.figure(figsize=(8, 6))
    plt.imshow(np.angle(Z_shifted[450:550, 450:550]))
    plt.colorbar()
    plt.title("Magnitude of the 2D Fourier Transform")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.tight_layout()
    plt.show()
