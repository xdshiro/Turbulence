import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Knot types
knots = [
    'standard_14', 'standard_16', 'standard_18',
    '30both', '30oneZ', 'optimized',
    'pm_03_z', '30oneX', '15oneZ',
    'trefoil_standard_12', 'trefoil_optimized'
]

# Stability data for 0.05 turbulence strength
stabilities_005 = [101 - 80, 101 - 70 + (70 - 66) / 2, 101 - 79 + 1 / 2,
                   101 - 85 + 3 / 2, 101 - 73, 100 - 40 + (40 - 31) / 2,
                   101 - 79 + 3 / 2, 101 - 58 + 4 / 2, 101 - 73 + 5 / 2,
                   16, 67.3]

# Confidence interval function
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Z-score for confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Parameters
n_samples = 200
confidence_level = 0.95

# Calculate confidence intervals for 0.05 turbulence strength
stabilities_005_delta = confidence_interval(stabilities_005, n_samples, confidence_level)

# Plot for stabilities_005
fig_005 = go.Figure()

# Add bar plot for 0.05 turbulence strength
fig_005.add_trace(go.Bar(
    x=knots,
    y=stabilities_005,
    error_y=dict(
        type='data',
        array=stabilities_005_delta,
        visible=True,
        color='black',  # Set error bar color to black for contrast
        thickness=2,    # Increase thickness
        width=4,        # Add wider caps for visibility
    ),
    name="0.05 Turbulence Strength",
    marker=dict(color='teal', line=dict(color='black', width=1)),  # Softer teal color
    width=0.5,  # Column width
))

# Layout customization
fig_005.update_layout(
    title="Stability of Optical Knots Under 0.05 Turbulence Strength (200 samples)",
    xaxis_title="Knot Type",
    yaxis_title="Knot Stability (%)",
    yaxis_range=[0, 100],
    font=dict(family='Arial', size=14),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        tickvals=list(range(len(knots))),
        ticktext=knots,
        tickangle=45,
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False
    ),
    showlegend=True,
    legend=dict(
        x=0.75,       # Adjust the x-position within the figure
        y=1,          # Move it down slightly
        bgcolor='rgba(255,255,255,0)',
        bordercolor='black',
        borderwidth=1
    ),
    bargap=0.15      # Control the gap between bars
)

# Display the plot
# fig_005.show()

# Save the plot to a file
fig_005.write_image("stability_plot_005.png", width=1000, height=650)  # Save as PNG