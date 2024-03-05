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
# file_names = [
#     'FF_x525_y735',
#     'FF_x525_y550',
#     'FF_x345_y745',
#     'FF_x300_y525'
# ]
df_weighted = []
df_middle = []
df_max = []
names = [
    '258',
'pi',
'pi_2',
]
# names = [
#     'H_258',
# 'new',
#
# ]
for name in names:
    # file_path_imag = f'{file_name}_imagEy.csv'
    # file_path_real = f'{file_name}_realEy.csv'
    file_path_imag = f'Slab\\Imag_FF_H_{name}.csv'
    file_path_real = f'Slab\\Real_FF_H_{name}.csv'
    file_path_imag = f'Slab\\Imag_FF_H_{name}.csv'
    file_path_real = f'Slab\\Real_FF_H_{name}.csv'
    # file_path_imag = f'Slab2\\Imag_THG_{name}.csv'
    # file_path_real = f'Slab2\\Real_THG_{name}.csv'


    # Reading CSV file, assuming the data starts directly without headers
    # Adjust 'skiprows' as needed if there are additional header lines
    data_imag = pd.read_csv(file_path_imag, skiprows=8, header=None)
    data_real = pd.read_csv(file_path_real, skiprows=8, header=None)
    # Extracting X, Y, and the fourth column (field amplitude)
    x_real = data_real[0].values
    y_real = data_real[1].values
    field_real = data_real[3].values  # Adjust the column index if necessary

    x_imag = data_imag[0].values
    y_imag = data_imag[1].values
    field_imag = data_imag[3].values  # Adjust the column index if necessary
    # Creating a 2D grid for interpolation
    xi = np.linspace(x_imag.min() * 0.99, x_imag.max() * 0.99, 200)
    yi = np.linspace(y_imag.min() * 0.99, y_imag.max() * 0.99, 200)
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
    print("Weighted average phase (int):", weighted_avg_phase / np.pi)
    df_weighted.append(weighted_avg_phase / np.pi)
    middle_indices = np.array(E.shape) // 2
    middle_phase = phase[middle_indices[0], middle_indices[1]]

    # Find the point of maximum amplitude
    max_amp_index = np.unravel_index(np.argmax(amplitude, axis=None), amplitude.shape)
    max_amp_phase = phase[max_amp_index]

    print("Phase at the middle of the field:", middle_phase / np.pi)
    df_middle.append(middle_phase / np.pi)
    print("Phase at the point of maximum amplitude:", max_amp_phase / np.pi)
    df_max.append(max_amp_phase / np.pi)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow((np.abs(E)), cmap='viridis', vmin=0, vmax=1)
    plt.title('Amp')
    plt.colorbar()

    # Filtered Spectrum
    plt.subplot(1, 2, 2)
    plt.imshow((np.angle(E)), cmap='jet', vmin=-np.pi, vmax=np.pi)
    plt.title('Phase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    continue
   
print(np.array(df_weighted) - df_weighted[-0])
print(np.array(df_middle) - df_middle[-0])
print(np.array(df_max) - df_max[-0])
