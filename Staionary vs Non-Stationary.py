#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:23:38 2024

@author: sadafmoaveninejad
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# Defining parameters for the signals
n_points = 3000
time = np.arange(n_points)

# Generating a stationary signal (fractional Gaussian noise)
H = 0.5
mean, std_dev = 0, 1
fGn = np.random.normal(mean, std_dev * (1 / (2 ** H)), size=n_points)

# Generating a non-stationary signal (fractional Brownian motion) by cumulatively summing the fGn
fBm = np.cumsum(fGn)

# Calculating variance for both signals over time for the lower plots
window_size = 3000
var_fGn = [np.var(fGn[max(i-window_size, 0):i+1]) for i in range(n_points)]
var_fBm = [np.var(fBm[max(i-window_size, 0):i+1]) for i in range(n_points)]

# Compute the variance (should be constant)
#fgn_var = [np.var(fGn)] * n_points
#fbm_var = [np.var(fBm

# Calculate the rolling means for fGn and fBm signals
mean_fGn = [np.mean(fGn[max(i-window_size, 0):i+1]) for i in range(n_points)]
mean_fBm = [np.mean(fBm[max(i-window_size, 0):i+1]) for i in range(n_points)]



import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Function to set the style of the axes with smaller fonts and reduced tick density
def set_ax_style(ax, title=None, x_label=None, y_label=None, ylim=None, show_xticks=True):
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=0.3)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold', labelpad=0.3)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold', labelpad=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Set the number of ticks based on a fixed number, rather than using all available ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # Adjust this number for more or fewer x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Adjust this number for more or fewer y-ticks
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=1, direction='out', length=2, pad=0.5)

    # Make tick labels bold
    for tick_label in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
        tick_label.set_fontweight('bold')

    
    # Only show x-ticks if this is a bottom subplot
    if not show_xticks:
        ax.set_xticks([])

# Adjust the figure to include two more subplots for the rolling means
fig, axs = plt.subplots(3, 2, figsize=(10, 14), dpi=200)

# Define colors and styles with increased linewidth
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Custom color scheme
color_stationary = '#1f77b4'  # 'navy'
color_nonstationary = '#ff7f0e'  # 'darkorange'
line_style = {'linestyle': '-', 'linewidth': 1.5}  # Increased linewidth

# Top left subplot
set_ax_style(axs[0, 0], title='Stationary Signal (fGn)', y_label='Amplitude', show_xticks=False)
axs[0, 0].plot(time, fGn, color=color_stationary, **line_style)

# Middle left subplot
set_ax_style(axs[1, 0], y_label='Variance', show_xticks=False)
axs[1, 0].plot(time, var_fGn, color=color_stationary, **line_style)
axs[1, 0].set_ylim([0, max(var_fGn) * 2.5])  # Adjust as needed

# Bottom left subplot
set_ax_style(axs[2, 0], x_label='Sample Index', y_label='Mean Value')
axs[2, 0].plot(mean_fGn, color=color_stationary, **line_style)
axs[2, 0].set_ylim([min(mean_fGn) * 2.5, max(mean_fGn) * 2.5])  # Adjust as needed

# Top right subplot
set_ax_style(axs[0, 1], title='Non-Stationary Signal (fBm)', show_xticks=False)
axs[0, 1].plot(time, fBm, color=color_nonstationary, **line_style)

# Middle right subplot
set_ax_style(axs[1, 1], show_xticks=False)
axs[1, 1].plot(time, var_fBm, color=color_nonstationary, **line_style)

# Bottom right subplot
set_ax_style(axs[2, 1], x_label='Sample Index')
axs[2, 1].plot(mean_fBm, color=color_nonstationary, **line_style)

plt.tight_layout()

# Optionally, save the figure
# plt.savefig('/path_to_save/fancy_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Set the parameters for the time series
n_points = 3000
H = 0.7  # Hurst parameter

# Generate a fractional Gaussian noise (fGn) sample
fgn = np.random.normal(size=n_points)
times = np.arange(n_points)

# Generate a fractional Brownian motion (fBm) by cumulative summation of fGn
fbm = np.cumsum(fgn)

# Compute the variance of fGn (should be constant)
fgn_var = [np.var(fgn)] * n_points

# Compute the variance of fBm (should change over time)
fbm_var = [np.var(fbm[:i+1]) for i in range(n_points)]

# Create the plots
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Plot fGn
axs[0, 0].plot(times, fgn)
axs[0, 0].set_title('Stationary Signal (fGn)')
axs[0, 0].set_xlabel('Sample Index')
axs[0, 0].set_ylabel('Amplitude')

# Plot fGn variance
axs[1, 0].plot(times, fgn_var)
axs[1, 0].set_title('Constant Variance')
axs[1, 0].set_xlabel('Sample Index')
axs[1, 0].set_ylabel('Variance')

# Plot fBm
axs[0, 1].plot(times, fbm)
axs[0, 1].set_title('Non-Stationary Signal (fBm)')
axs[0, 1].set_xlabel('Sample Index')
axs[0, 1].set_ylabel('Amplitude')

# Plot fBm variance
axs[1, 1].plot(times, fbm_var)
axs[1, 1].set_title('Varying Variance')
axs[1, 1].set_xlabel('Sample Index')
axs[1, 1].set_ylabel('Variance')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()
