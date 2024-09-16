
# optimized:  [95.66666667 67.33333333 37.         21.66666667  7.66666667]
# standard_1.15:  [43.         16.          5.33333333  1.66666667  0.66666667]
# dennis: [77.33333333 41.         11.66666667  5.66666667  2.33333333]
optimized = 67.3
standard_115 = 16
dennis = 41
math_5 = 30
math_many = 38
manth_many_smaller = 50
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data for different knots
knots = ['Optimized', 'Standard 115', 'Dennis', 'Math 5', 'Math Many', 'Math Many Smaller']
stability_values = [67.3, 16, 41, 30, 38, 50]

# Corresponding sample sizes for each knot
samples = [300, 300, 300, 200, 200, 200]

# Confidence Interval Function
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Calculate the z-value dynamically based on confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Calculate the confidence intervals for each knot stability value using corresponding sample sizes
ci = [confidence_interval([stability], sample)[0] for stability, sample in zip(stability_values, samples)]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(knots, stability_values, yerr=ci, capsize=5, color='skyblue', edgecolor='blue', alpha=0.7)

# Labels and title
plt.xlabel('Knot Types')
plt.ylabel('Stability in Turbulence (%)')
plt.title('Knot Stability at Turbulence Strength 0.05 with Confidence Intervals')

# Show the plot
plt.tight_layout()
plt.show()