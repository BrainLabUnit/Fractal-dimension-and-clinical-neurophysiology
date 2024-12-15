#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:45:47 2023

@author: sadafmoaveninejad
"""

import numpy as np
import pandas as pd
from functions_Higuchi import compute_FD, wsc, generate_fbm, generate_fgn, illustrate_original_timeseries, takagi_landsberg
from functions_Higuchi import illustrate_higuchi_parameters, illustrate_higuchi_Lmk_values, plot_L_and_FD, plot_FD_vs_Kmax


#%% ------ 1- Synthetic data ------

# ---- Parameters ------
fs = 30#256#128#60
t = np.linspace(0, 1 - 1/fs, fs)
N = len(t)

lamda = 5
K_max = int((N/2))
FD = 1.5
H = 2 - FD  # Convert fractal dimension to Hurst exponent

# ----- time series -------
# ---- WCF ----
wcf_series = wsc(fs, lamda, K_max, H)
# --- fGN, fBM ---
fbm_series = generate_fbm(N, H)
fgn_series = generate_fgn(N, H)


#%%
time_series = wcf_series#fbm_series#fgn_series #fgn_series #

#%% Plot Time series 
#K_max=5
k_list = np.linspace(2, K_max, 5, dtype=int)  # Create 3 evenly spaced values from 1 to K_max
#k_list = np.linspace(2, K_max, K_max-1, dtype=int)  # Create 3 evenly spaced values from 1 to K_max
# ----- Plot -----
illustrate_original_timeseries(time_series, FD)

#%% 
N = len(time_series)
m, k = 3,14
total_num_points = (N-1)
num_segments_m = int((N-m)/k)
num_segments_N = int((N-1)/k)
normalization = num_segments_N / num_segments_m 

# --------------- Xm(k), m,k ------------------
cases = [(1,5), (2,5), (3,5), (4,5), (5,5)] #(m,k)
cases = [(1,6), (2,6), (3,6), (4,6), (5,6), (6,6)] #(m,k)
cases = [(1,4), (2,4), (3,4), (4,4)] #(m,k)
cases = [(1,3), (2,3), (3,3)]#,(8,15)] #(m,k)
cases = [(1,2), (2,2)] #(m,k)
cases = [(1,15),(5,15), (8,15),(15,15)] #(m,k)
cases = [(1,29),(2,29), (3,29),(29,29)] #(m,k)
cases = [(1,60),(15,60), (30,60),(60,60)]
cases = [(8,15)]

illustrate_higuchi_parameters(time_series, cases)

# --------------- Lm(k) ------------------
k_list = [1,2,3,14,15]#[1,6,9,15]#
illustrate_higuchi_Lmk_values(time_series, k_list)

# --------------- L(k), FD ---------------
k_list = [15]#[1,6,9,15]#
plot_L_and_FD(time_series, k_list)

#---------------- FD vs K_max ------------
Kmax_list = np.linspace(2, K_max, K_max, dtype=int)
FD_values = plot_FD_vs_Kmax(Kmax_list, time_series)

#%% --------------- different FD -----
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_FD_vs_Kmax_superimposed(FD_list, K_max, vertical_lines, N, box_k_min, box_k_max):
    fig, ax = plt.subplots(figsize=(10, 6),dpi=200)
    Kmax_list = np.linspace(2, K_max, K_max - 1, dtype=int)  # Corrected length of Kmax_list

    # Colors for each FD value
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'red']

    # Add a shaded box between box_k_min and box_k_max
    ax.axvspan(box_k_min, box_k_max, color='lightgray', alpha=0.5,)

    # Plot each FD curve
    for i, (FD, H) in enumerate(zip(FD_list, H_list)):
        # Generate time series
        time_series = wsc(fs, lamda, K_max, H)
        
        # Calculate FD values
        FD_values = [compute_FD(time_series, k) for k in Kmax_list]
        
        # Plot FD vs Kmax
        ax.plot(Kmax_list, FD_values, '-o', color=colors[i], markerfacecolor=colors[i],
                markeredgewidth=2, markersize=6, linewidth=2, label=f'$FD_{{th}}={FD}$')
        
        # Annotate FD values at each vertical line
        for idx, k in enumerate(vertical_lines):  # Use enumerate to get index
            if k in Kmax_list:
                k_index = list(Kmax_list).index(k)  # Find the index corresponding to the vertical line's K value
                fd_at_k = FD_values[k_index]  # Get FD value at the K value
                if idx == 0:
                    print(i)
                    # Alternate annotation direction for different FD values
                    if i % 2 == 0:  # Even FD index -> annotate from the right
                        xytext_offset = (k + 2, fd_at_k - 0.05)
                    else:  # Odd FD index -> annotate from the left
                        xytext_offset = (k - 10, fd_at_k - 0.05)
                else:
                    xytext_offset = (k + 1, fd_at_k )#- 0.05

                ax.annotate(f'$FD_{{H}}={fd_at_k:.2f}$', xy=(k, fd_at_k), xytext=xytext_offset,
                            textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color=colors[i]),
                            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[i], facecolor='#e6eff7'))

    # Add vertical lines at specified K values
    for k in vertical_lines:
        ax.axvline(x=k, linestyle='--', linewidth=2, alpha=0.7,color='gray')

    # Set axis labels
    ax.set_xlabel('K', fontsize=16, fontweight='bold')
    ax.set_ylabel('HFD', fontsize=16, fontweight='bold')  # Updated y-axis label
    ax.set_ylim([1, 2])
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Add legend at the top-right corner with title inside the box
    legend = ax.legend(fontsize=12, frameon=True, shadow=True, fancybox=True, loc="upper right")
    legend.set_title('Theoretical FD ($FD_{th}$)', prop={'size': 10, 'weight': 'bold'})

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=2, colors='#333333')

    # Make tick labels bold
    for tick_label in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
        tick_label.set_fontweight('bold')

    plt.tight_layout()
    plt.show()

# Example Parameters
fs = 128  # Sampling frequency
t = np.linspace(0, 1 - 1/fs, fs)  # Time array
N = len(t)
lamda = 10
K_max = int((N / 2))
FD_list = [1.4, 1.5, 1.6]  # Different FD values to plot
H_list = [2 - fd for fd in FD_list]  # Convert FD to Hurst exponent

# Specify K values for vertical lines
vertical_lines = [3, 24]  # Example vertical line positions

# Specify K limits for the shaded region
box_k_min = 2
box_k_max = 10

# Call the Plotting Function
plot_FD_vs_Kmax_superimposed(FD_list, K_max, vertical_lines, N, box_k_min, box_k_max)


