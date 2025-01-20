import numpy as np
import matplotlib.pyplot as plt
from aotools.turbulence import phasescreen
from scipy.special import laguerre
from aotools import opticalpropagation
from aotools.turbulence.phasescreen import ft_phase_screen as ps
from aotools.turbulence.phasescreen import ft_sh_phase_screen as psh
# from aotools.turbulence.phasescreen import ft_phase_screen as psh
import math
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg

import plotly.graph_objects as go

from functions.all_knots_functions import *

import os
import pickle
import csv
import json
from tqdm import trange
import itertools
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

fopts = 24  # Font size for labels
tick_fopts = 24  # Font size for tick numbers
plt.rc('font', family='Times New Roman')


def plotDots_foils_paper_by_phi(dots, dots_bound=None, show=True, size=15, width=185, fig=None, save=None, reso=100):
	colorLine = 'white'
	
	# Define angle sections (phi ranges)
	z1 = np.linspace(-1 / 4 * np.pi, 1 / 4 * np.pi, reso)
	z2 = np.linspace(1 / 4 * np.pi, 3 / 4 * np.pi, reso)
	z31 = np.linspace(3 / 4 * np.pi, np.pi, reso // 2)
	z32 = np.linspace(-np.pi, -3 / 4 * np.pi, reso // 2)
	z4 = np.linspace(-3 / 4 * np.pi, -1 / 4 * np.pi, reso)
	
	# Compute the angle phi for each dot
	phi = np.arctan2(dots[:, 1], dots[:, 0])  # arctan2(y, x) gives angles in [-pi, pi]
	
	# Separate dots into 4 sections based on phi
	dots_z1 = dots[(phi >= -1 / 4 * np.pi) & (phi < 1 / 4 * np.pi)]
	dots_z2 = dots[(phi >= 1 / 4 * np.pi) & (phi < 3 / 4 * np.pi)]
	dots_z31 = dots[(phi >= 3 / 4 * np.pi) | (phi < -3 / 4 * np.pi)]  # Handles wrap-around
	dots_z4 = dots[(phi >= -3 / 4 * np.pi) & (phi < -1 / 4 * np.pi)]
	
	# Assign colors
	colors = {
		"z1": 'red',
		"z2": 'blue',
		"z31": 'green',
		"z4": 'purple'
	}
	
	# Plot each section with corresponding color
	if fig is None:
		fig = pl.plot_3D_dots_go(dots_z1, marker={'size': size, 'color': colors['z1'],
		                                          'line': dict(width=width, color=colorLine)})
	else:
		pl.plot_3D_dots_go(dots_z1, fig=fig, marker={'size': size, 'color': colors['z1'],
		                                             'line': dict(width=width, color=colorLine)})
	
	pl.plot_3D_dots_go(dots_z2, fig=fig, marker={'size': size, 'color': colors['z2'],
	                                             'line': dict(width=width, color=colorLine)})
	pl.plot_3D_dots_go(dots_z31, fig=fig, marker={'size': size, 'color': colors['z31'],
	                                              'line': dict(width=width, color=colorLine)})
	pl.plot_3D_dots_go(dots_z4, fig=fig, marker={'size': size, 'color': colors['z4'],
	                                             'line': dict(width=width, color=colorLine)})
	
	# Set up the bounding box
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


def plot_shifted_paper_grid_spectrum(spectrum, l1, l2, p1, p2, figsize=(8.5, 6),
                                     map=plt.cm.gist_earth, title='',

                                     grid=True, axis_equal=False, xname='l', yname='p',
                                     xlim=None, ylim=None, every_ticks=False):
	"""
	Plot the spectrum with grid shifted by 0.5 and custom settings.

	Parameters:
		spectrum: 2D numpy array to plot.
		l1, l2: Start and end points for the x-axis.
		p1, p2: Start and end points for the y-axis.
		map: Colormap for the plot.
		title: Title of the plot.
		xyLabelFontSize: Font size for axis labels.
		ticksFontSize: Font size for tick numbers.
		grid: Boolean, whether to show the grid.
		axis_equal: Boolean, whether to set equal aspect ratio.
		xname, yname: Names for the x and y axes.
		xlim, ylim: Optional x and y axis limits.
	"""
	
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	# Create shifted grid
	x = np.arange(l1 - 0.5, l2 + 1.5, 1)  # Shift grid in x by 0.5
	y = np.arange(p1 - 0.5, p2 + 1.5, 1)  # Shift grid in y by 0.5
	# x = np.arange(l1, l2 + 1, 1)  # Shift grid in x by 0.5
	# y = np.arange(p1, p2 + 1, 1)  # Shift grid in y by 0.5
	
	# Create the figure and axis
	# fig, ax = plt.subplots(figsize=(13.5, 6))
	fig, ax = plt.subplots(figsize=figsize)
	# print(np.sum((np.abs(spectrum)/100)**2))
	# Plot the spectrum with shifted extent
	image = ax.imshow(np.abs(spectrum).T / 100,
	                  interpolation='none', cmap=map,
	                  origin='lower', aspect='auto', vmin=0,
	                  extent=[x[0], x[-1], y[0], y[-1]])
	
	# Add colorbar
	cbr = plt.colorbar(image, ax=ax, shrink=0.8, pad=0.02, fraction=0.1)
	cbr.ax.tick_params(labelsize=ticksFontSize)  # Colorbar tick font size
	
	# Axis ticks and labels
	if every_ticks:
		ax.set_xticks(np.arange(l1, l2 + 1, 1))  # Main x-axis ticks
	else:
		ax.set_xticks(np.arange(l1, l2 + 1, 2))  # Main x-axis ticks
	ax.set_yticks(np.arange(p1, p2 + 1, 1))  # Main y-axis ticks
	ax.set_xlabel(xname, fontsize=xyLabelFontSize, fontstyle='italic')
	ax.set_ylabel(yname, fontsize=xyLabelFontSize, fontstyle='italic')
	plt.xticks(fontsize=ticksFontSize)
	plt.yticks(fontsize=ticksFontSize)
	
	# Grid settings with minor ticks shifted by 0.5
	# ax.grid(grid, which='major', linestyle='--', alpha=0.7)
	ax.xaxis.set_minor_locator(plt.MultipleLocator(1), )  # Minor ticks every 1 unit
	ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
	ax.grid(True, which='minor', linestyle='--', alpha=0.7)
	
	# Minor grid alignment shift
	ax.set_xticks(np.arange(l1 - 0.5, l2 + 1.5, 1), minor=True)  # Shifted minor x ticks
	ax.set_yticks(np.arange(p1 - 0.5, p2 + 1.5, 1), minor=True)  # Shifted minor y ticks
	# Minor ticks are disabled, but minor grid remains
	ax.tick_params(axis='x', which='minor', length=0, width=0, direction='in', labelsize=0)
	ax.tick_params(axis='y', which='minor', length=4, width=1, direction='in', labelsize=0)
	# Plot title
	# plt.title(title, fontweight="bold", fontsize=26)
	
	# Optional settings
	if axis_equal:
		ax.set_aspect('equal', adjustable='box')
	if xlim is not None:
		ax.set_xlim(xlim[0], xlim[1])
	if ylim is not None:
		ax.set_ylim(ylim[0], ylim[1])
	
	# Adjust layout and display the plot
	plt.tight_layout()
	plt.show()


def plotDots_foils_paper_by_indices(dots, indices, dots_bound=None, show=True, size=15, width=185, fig=None, save=None):
	"""
	Plot dots in 3D space with coloring based on a sequence of indices dividing the array into 4 parts.

	Parameters:
		dots: numpy array of shape (n, 3) representing 3D points.
		indices: list of 4 integers defining the division points in the array.
		dots_bound: Optional, bounding box points for the plot.
		show: Boolean, whether to display the plot.
		size: Marker size for the dots.
		width: Line width for the markers.
		fig: Optional, existing plotly figure to add to.
		save: Optional, file name to save the plot as HTML.

	Returns:
		fig: Plotly figure object.
	"""
	colorLine = 'white'
	
	# Sort indices to ensure they divide the array correctly
	indices = sorted(indices)
	
	# Split the dots array based on the provided indices
	dots_part1 = dots[indices[0]:indices[1]]  # From first to second index
	dots_part2 = dots[indices[1]:indices[2]]  # Between second and third index
	dots_part3 = dots[indices[2]:indices[3]]  # Between third and fourth index
	dots_part4 = np.vstack((dots[:indices[0]], dots[indices[3]:]))  # Combine [0:indices[0]] and [indices[3]:]
	
	# Assign requested colors to each section
	colors = ['#ff0000', '#007dff', '#ff9900', '#19ff19']
	colors = ['#660000', '#000099', '#ffff00', '#134a0d']
	# red blue orange greed
	colors = ['#ff0000', '#000099', '#ff9900', '#134a0d']
	# blue orange greed red
	colors = ['#000099', '#ff9900', '#134a0d', '#ff0000']
	colors = ['#000099', '#ffcc00', '#166d13', '#ff0000']
	colors = ['#000099', '#ffb200', '#166d13', '#ff0000']
	dots_parts = [dots_part1, dots_part2, dots_part3, dots_part4]
	
	# Plot each section with corresponding color
	if fig is None:
		fig = pl.plot_3D_dots_go(dots_parts[0], marker={'size': size, 'color': colors[0],
		                                                'line': dict(width=width, color=colorLine)})
	for i in range(1, len(dots_parts)):
		pl.plot_3D_dots_go(dots_parts[i], fig=fig, marker={'size': size, 'color': colors[i],
		                                                   'line': dict(width=width, color=colorLine)})
	
	# Set up the bounding box
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	camera = dict(
		eye=dict(x=0, y=0, z=2),  # Top-down view
		up=dict(x=0, y=-1, z=0),  # Rotate around z-axis by 90 degrees
		center=dict(x=0, y=0, z=0)  # Keep the center of the plot unchanged
	)
	fig.update_layout(scene_camera=camera)
	
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


def plot_black_dots_paper(dots, dots_bound=None, size=15, width=185, show=True, fig=None, save=None,
                          general_view=False):
	"""
	Plot dots in 3D space as black dots.

	Parameters:
		dots: numpy array of shape (n, 3) representing 3D points.
		size: Marker size for the dots.
		width: Line width for the markers.
		show: Boolean, whether to display the plot.
		fig: Optional, existing plotly figure to add to.
		save: Optional, file name to save the plot as HTML.

	Returns:
		fig: Plotly figure object.
	"""
	colorLine = 'white'
	# Set up default figure if none is provided
	if fig is None:
		fig = pl.plot_3D_dots_go(dots, marker={
			'size': size,
			'color': 'black',
			'line': dict(width=width, color=colorLine)  # Border color for the dots
		})
	else:
		pl.plot_3D_dots_go(dots, fig=fig, marker={
			'size': size,
			'color': 'black',
			'line': dict(width=width, color=colorLine)
		})
	if dots_bound is None:
		dots_bound = dots
	pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
	if general_view:
		scale_zoom = 2.45
		camera = dict(
			eye=dict(x=1.25 * scale_zoom, y=1.25 * scale_zoom, z=1.25 * scale_zoom),
			up=dict(x=0, y=0, z=1),
			center=dict(x=0, y=0, z=0)
		)
	else:
		camera = dict(
			eye=dict(x=0, y=0, z=2),  # Top-down view
			up=dict(x=0, y=-1, z=0),  # Rotate around z-axis by 90 degrees
			center=dict(x=0, y=0, z=0)  # Keep the center of the plot unchanged
		)
	fig.update_layout(scene_camera=camera)
	# Save or show the figure
	if save is not None:
		fig.write_html(save)
	if show:
		fig.show()
	
	return fig


def plot_field_both_paper(E, extend=None):
	# Set Times New Roman as the font globally

	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts

	fig, ax = plt.subplots(1, 2, figsize=(13.5, 6))

	# Plot |Amplitude|
	im0 = ax[0].imshow(np.abs(E).T, extent=extend, cmap="Blues_r", interpolation='nearest')
	cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.01)
	cbar0.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar0.set_ticks([0, 1])  # Custom ticks for |Amplitude|
	ax[0].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[0].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[0].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size

	# Plot Phase
	# im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='gray', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.01)
	cbar1.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar1.set_ticks([-np.pi, 0, np.pi])  # Custom ticks for Phase
	cbar1.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])  # Use LaTeX for Phase ticks
	ax[1].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[1].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[1].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size

	# Adjust layout
	plt.tight_layout()

	# Display the figure
	plt.show()

def plot_field_both_paper_old_version(E, extend=None):
	# Set Times New Roman as the font globally
	
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	
	fig, ax = plt.subplots(1, 2, figsize=(13.5, 6))
	
	# Plot |Amplitude|
	im0 = ax[0].imshow(np.abs(E).T, extent=extend, cmap='hot', interpolation='nearest')
	cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.01)
	cbar0.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar0.set_ticks([0, 1])  # Custom ticks for |Amplitude|
	ax[0].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[0].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[0].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size
	
	# Plot Phase
	# im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	im1 = ax[1].imshow(np.angle(E).T, extent=extend, cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
	cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.01)
	cbar1.ax.tick_params(labelsize=ticksFontSize)  # Set colorbar tick font size
	cbar1.set_ticks([-np.pi, 0, np.pi])  # Custom ticks for Phase
	cbar1.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])  # Use LaTeX for Phase ticks
	ax[1].set_xlabel('x (mm)', fontsize=xyLabelFontSize)  # X-axis label
	ax[1].set_ylabel('y (mm)', fontsize=xyLabelFontSize)  # Y-axis label
	ax[1].tick_params(axis='both', labelsize=ticksFontSize)  # Set axis tick font size
	
	# Adjust layout
	plt.tight_layout()
	
	# Display the figure
	plt.show()


def plot_confusion_matrix(cm, class_labels, label='', figsize=(11, 9.5), cmap="Blues",
                          annot_fontsize=fopts):
	"""
	Plot the confusion matrix with custom font sizes and class labels.

	Parameters:
		cm: 2D numpy array
			The confusion matrix to be plotted.
		class_labels: list
			List of class labels to use as x and y ticks.
		figsize: tuple
			Size of the plot figure.
		cmap: str
			Colormap for the heatmap.
			:param annot_fontsize: size of the confusion numbers
	"""
	
	xyLabelFontSize = fopts
	ticksFontSize = tick_fopts
	# Create the figure
	plt.figure(figsize=figsize)
	vmin = 0
	vmax = 100
	# Plot the heatmap
	ax = sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels,
	            cmap=cmap, annot_kws={"size": annot_fontsize}, vmin=vmin, vmax=vmax,
                cbar_kws={"shrink": 0.9, "aspect": 20,
                          "ticks": np.linspace(vmin, vmax, 5),
                          "pad": 0.01,
                          })

	cbar = plt.gcf().axes[-1]  # Get the colorbar axis
	cbar.tick_params(labelsize=ticksFontSize)  # Set the font size for colorbar ticks
	cbar.set_ylabel("Accuracy (%)", fontsize=xyLabelFontSize)  # Set label and font size
	# Add labels and customize font sizes
	plt.xlabel('Predicted Knots', fontsize=xyLabelFontSize)
	plt.ylabel('True Knots', fontsize=xyLabelFontSize)
	plt.xticks(fontsize=ticksFontSize)
	plt.yticks(fontsize=ticksFontSize)
	ax.set_aspect('equal')
	plt.title(label, fontsize=xyLabelFontSize, fontweight="bold")
	# plt.subplots_adjust(right=0.95)  # Shrink the right margin
	plt.tight_layout()
	plt.show()
