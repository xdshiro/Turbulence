# stabilities_025 = [0 + 1 / 2, 3 + 3 / 2, 1 + 2 / 2,
#                    2 + 1 / 2, 3 + 1 / 2, 7 + 5 / 2,
#                    3 + 1 / 2, 8 + 1 / 2, 1 + 0 / 2,
#                    1 + 0 / 2, 3 + 3 / 2]
#
# stabilities_005 = [101 - 80, 101 - 70 + (70 - 66) / 2, 101 - 79 + 1 / 2,
#                    101 - 85 + 3 / 2, 101 - 73, 100 - 40 + (40 - 31) / 2,
#                    101 - 79 + 3 / 2, 101 - 58 + 4 / 2, 101 - 73 + 5 / 2,
#                    16, 67.3]
#
# stabilities_015 = [101 - 95 + 3 / 2, 101 - 92 + 2 / 2, 3,
#                    4 + 3 / 2, 3 + 1 / 2, 100 - 76 + (76 - 66) / 2,
#                    2 + 2 / 2, 101 - 86 + 6 / 2, 101 - 95 + 3 / 2,
#                    1.6, 21.6]


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

knots = [
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'T1', 'T2'
]
# Stability data
stabilities_005 = [101 - 80, 101 - 70 + (70 - 66) / 2, 101 - 81 + 0 / 2,
                   101 - 85 + 4 / 2, 101 - 75, 100 - 40 + (40 - 32) / 2,
                   101 - 79 + 2 / 2, 101 - 58 + 4 / 2, 101 - 73 + 4 / 2,
                   16, 67.3333333]

stabilities_015 = [101 - 95 + 2 / 2, 101 - 92 + 2 / 2, 4,
                   4 + 2 / 2, 5 + 0 / 2, 100 - 79 + (76 - 66) / 2,
                   2 + 2 / 2, 101 - 86 + 6 / 2, 101 - 95 + 2 / 2,
                   1.6666666, 21.666666]

stabilities_025 = [1, 3 + 2 / 2, 1 + 2 / 3,
                   2 + 2 / 3, 3 + 0 / 2, 5 + 6 / 2,
                   2 + 1 / 3, 4 + 0 / 2, 1 + 1 / 3,
                   1 + 0 / 2, 3 + 2 / 2]

# Confidence interval function
def confidence_interval(p, n, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)  # Z-score for confidence level
    p = np.array(p) / 100  # Convert percentage to proportion
    se = np.sqrt((p * (1 - p)) / n)  # Standard error
    ci = z * se  # Confidence interval
    return ci * 100  # Convert back to percentage

# Parameters
n_samples = 300
confidence_level = 0.95

# Calculate confidence intervals
stabilities_005_delta = confidence_interval(stabilities_005, n_samples, confidence_level)
stabilities_015_delta = confidence_interval(stabilities_015, n_samples, confidence_level)
stabilities_025_delta = confidence_interval(stabilities_025, n_samples, confidence_level)

# Define the space between columns by adjusting the position of each column
x_shift = 0.27  # Shift factor for column spacing
width = 0.25
group_shift = 0.9
# Plot for stabilities_005 with more space for all three columns
fig_005 = go.Figure()

# Add bar plot for 0.05 turbulence strength with more spacing
fig_005.add_trace(go.Bar(
    x=[i * group_shift - x_shift for i in range(len(knots))],
    y=stabilities_005,
    error_y=dict(
        type='data',
        array=stabilities_005_delta,
        visible=True,
        color='black',  # Set error bar color to black for contrast
        thickness=2,    # Increase thickness
        width=4,        # Add wider caps for visibility
    ),
    name=r"$\sigma_R^2=0.05$",
    marker=dict(color='teal', line=dict(color='black', width=1)),  # Softer teal color
    width=width,  # Narrower column width to create more space between columns
))

# Add bar plot for 0.15 turbulence strength with more spacing
fig_005.add_trace(go.Bar(
    x=[i * group_shift for i in range(len(knots))],
    y=stabilities_015,
    error_y=dict(
        type='data',
        array=stabilities_015_delta,
        visible=True,
        color='black',  # Set error bar color to black for contrast
        thickness=2,    # Increase thickness
        width=4,        # Add wider caps for visibility
    ),
    name=r"$\sigma_R^2=0.15$",
    marker=dict(color='salmon', line=dict(color='black', width=1)),  # Softer salmon color
    width=width,  # Narrower column width to create more space between columns
))

# Add bar plot for 0.25 turbulence strength with more spacing
fig_005.add_trace(go.Bar(
    x=[i * group_shift + x_shift for i in range(len(knots))],
    y=stabilities_025,
    error_y=dict(
        type='data',
        array=stabilities_025_delta,
        visible=True,
        color='black',
        thickness=2,
        width=4,
    ),
    name=r"$\sigma_R^2=0.25$",
    marker=dict(color='lightblue', line=dict(color='black', width=1)),
    width=width,  # Narrower column width to create more space between columns
))

fig_005.update_layout(
    # title="Stability of Optical Knots Under Different Turbulence Strengths",
    xaxis_title="Knot Type",
    yaxis_title="Recovered Knots(%)",
    yaxis_range=[0, 100],
    font=dict(family='Arial', size=22, color='black'),  # Set all font colors to black
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        tickvals=[i * group_shift for i in range(len(knots))],  # Align tick positions with bar groups
        ticktext=knots,  # Correctly assign tick labels
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
    bargap=0.15,       # Adjust spacing within groups
    bargroupgap=0.2    # Adjust spacing between groups
,
    margin=dict(
        l=50,   # Left margin
        r=10,   # Right margin
        t=5,   # Top margin
        b=50    # Bottom margin
    ),
    width=1000,         # Set the width of the figure
    height=650         # Set the height of the figure
)



# Display the plot
# fig_005.show()



# Save the plot to a file
fig_005.write_image("stability_plot_005_015_025.png") #, width=1000, height=650)  # Save as PNG
# fig_005.write_image("stability_plot_005.pdf")  # Save as PDF