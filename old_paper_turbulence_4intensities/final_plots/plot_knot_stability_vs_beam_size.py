import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data
stability_beam_size = [42, 54, 68, 69, 67.33, 51, 36]
beam_size = [4, 4.5, 5, 5.5, 6, 6.5, 7]
number_of_samples = 200

# Confidence Interval Function
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Calculate the confidence intervals
ci = confidence_interval(stability_beam_size, number_of_samples)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(beam_size, stability_beam_size, label='Knot Stability', color='b', marker='o')

# Fill the area between the confidence interval
plt.fill_between(beam_size, 
                 np.array(stability_beam_size) - ci, 
                 np.array(stability_beam_size) + ci, 
                 color='blue', alpha=0.2, label='Confidence Interval (95%)')

# Labels and title
plt.xlabel('Beam Size')
plt.ylabel('Knot Stability in Turbulence (%)')
plt.title('Knot Stability vs Beam Size')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)
# Show the plot
plt.show()