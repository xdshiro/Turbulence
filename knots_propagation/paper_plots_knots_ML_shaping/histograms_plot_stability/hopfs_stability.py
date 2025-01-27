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

# Knot types
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
    from scipy.stats import norm
    z = norm.ppf((1 + confidence_level) / 2)  # Z-score
    p = np.array(p) / 100.0                  # Convert to fraction
    se = np.sqrt((p * (1 - p)) / n)          # Standard error
    ci = z * se                              # Margin of error
    return ci * 100                          # Back to %

# Calculate confidence intervals
n_samples = 300
stabilities_005_delta = confidence_interval(stabilities_005, n_samples)
stabilities_015_delta = confidence_interval(stabilities_015, n_samples)
stabilities_025_delta = confidence_interval(stabilities_025, n_samples)

# Spacing parameters
x_shift = 0.27
width = 0.25
group_shift = 0.9

# Define three "Blues" shades (ColorBrewer Blues for 3 categories)
# You can pick any you like, or lighten/darken them as needed
blues_colors = ["#deebf7", "#9ecae1", "#3182bd"]

# Build figure
fig_005 = go.Figure()

# ---------- 0.05 Turbulence ----------
fig_005.add_trace(go.Bar(
    x=[i * group_shift - x_shift for i in range(len(knots))],
    y=stabilities_005,
    error_y=dict(
        type='data',
        array=stabilities_005_delta,
        visible=True,
        color='black',
        thickness=2,
        width=4,
    ),
    name=r"$\sigma_R^2=0.05$",
    # name=r"$\sigma_R^2=0.05$",
    marker=dict(
        color=blues_colors[0],
        line=dict(color='black', width=1)  # Black border for each bar
    ),
    width=width,
    # Add the values on top of each bar:
    text=[f"{val:.1f}" for val in stabilities_005],  # Format to 1 decimal place
    textposition='outside',        # Place text above the bar
    textfont=dict(size=14, color='black')
))

# ---------- 0.15 Turbulence ----------
fig_005.add_trace(go.Bar(
    x=[i * group_shift for i in range(len(knots))],
    y=stabilities_015,
    error_y=dict(
        type='data',
        array=stabilities_015_delta,
        visible=True,
        color='black',
        thickness=2,
        width=4,
    ),
    name=r"$\sigma_R^2=0.15$",
    marker=dict(
        color=blues_colors[1],
        line=dict(color='black', width=1)
    ),
    width=width,
    text=[f"{val:.1f}" for val in stabilities_015],
    textposition='outside',
    textfont=dict(size=14, color='black')
))

# ---------- 0.25 Turbulence ----------
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
    marker=dict(
        color=blues_colors[2],
        line=dict(color='black', width=1)
    ),
    width=width,
    text=[f"{val:.1f}" for val in stabilities_025],
    textposition='outside',
    textfont=dict(size=14, color='black')
))

fig_005.update_layout(
    xaxis_title="Knot Type",
    yaxis_title="Recovered Knots(%)",
    yaxis_range=[0, 100],
    font=dict(family='Arial', size=20, color='black'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        tickvals=[i * group_shift for i in range(len(knots))],
        ticktext=knots,
        tickangle=0,
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False,
        # Draw black frame around X-axis:
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True  # Mirror means we also get a top axis line
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray',
        zeroline=False,
        # Draw black frame around Y-axis:
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True
    ),
    showlegend=True,
    legend=dict(
        x=0.867,
        y=0.99,
        bgcolor='rgba(255,255,255,0)',
        bordercolor='black',
        borderwidth=1
    ),
    bargap=0.15,
    bargroupgap=0.2,
    margin=dict(l=50, r=10, t=5, b=50),
    width=1000,
    height=650
)

# Show figure (uncomment if running interactively)
fig_005.show()

# Save the plot to a file
# fig_005.write_html("stability_plot_005_015_025_2.html")