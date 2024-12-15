#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:33:13 2023

@author: sadafmoaveninejad
"""

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

#%% ----- Signals ----- 

def wsc(N, lamda, M, H):
    t = np.linspace(0, 1, N)
    x_wsc = np.zeros(N)
    for i in range(N):
        x_wsc[i] = np.sum((lamda ** -((np.arange(M+1) * H))) * np.cos(2 * np.pi * (lamda ** (np.arange(M+1))) * t[i]))
    return x_wsc

def generate_fbm(N, H):
    # Generate fractional Brownian motion
    f = FBM(n=N-1, hurst=H, length=1, method='daviesharte')
    fbm_series = f.fbm()  # This is the fBm time series
    return fbm_series

def generate_fgn(N, H):
    # Generate fractional Gaussian noise
    f = FBM(n=N, hurst=H, length=1, method='daviesharte')  # N+1 because differencing reduces the length by 1
    fgn_series = f.fgn()  # This is the fGn time series
    return fgn_series

def takagi_landsberg(N, H, K_max):
    t = np.linspace(0, 1, N)
    series = np.zeros_like(t)
    for k in range(K_max):
        series += (1 / 2) ** (k * H) * np.abs(np.sin(2 ** k * np.pi * t))
    return series


#%% ----- Higuchi Parameters - Part II:  L_m(k) ----- 

import numpy as np
from scipy.stats import linregress

def Lmk(time_series, m, k):
    N = len(time_series)
    num_segments_m = int(np.floor((N - m) / k))
    num_segments_N = int(np.floor((N - 1) / k))

    if num_segments_m < 1:
        return None

    summation = sum(abs(time_series[m + i * k - 1] - time_series[m + (i - 1) * k - 1]) for i in range(1, num_segments_m + 1))
    Ng = (N - 1) / (num_segments_m * k)
    Lmk_value = (summation * Ng) / k

    return Lmk_value

def mean_Lk(time_series, k):
    L_values = [Lmk(time_series, m, k) for m in range(1, k + 1)]
    L_values = [l for l in L_values if l is not None]
    return np.mean(L_values) if L_values else 0

def compute_FD(time_series, max_k=5):
    k_values = list(range(1, max_k + 1))
    L_values = [mean_Lk(time_series, k) for k in k_values]

    valid_k_values = [k for k, l in zip(k_values, L_values) if l > 0]
    valid_L_values = [l for l in L_values if l > 0]

    if not valid_L_values:
        print("Insufficient valid L(k) values.")
        return None

    log_k_values = np.log10(1 / np.array(valid_k_values))
    log_L_values = np.log10(valid_L_values)

    slope, _, _, _, _ = linregress(log_k_values, log_L_values)
    if max_k ==1:
        return slope
    else:
        return truncate(slope, 2)  # Truncate the slope to two decimal places

def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("Decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("Decimal places must be non-negative.")
    elif decimals == 0:
        return int(number)

    factor = 10.0 ** decimals
    return int(number * factor) / factor

#%% ----- Higuchi Parameters - Part I:  (k,m) ----- 
def illustrate_original_timeseries(time_series, FD=None):
    N = len(time_series)
    plt.figure(figsize=(8, 6),dpi=150)
    # Plot the original time series with a solid line
    plt.plot(np.arange(1, N+1), time_series, color='black', lw=3, label='Original Time Series')
    # Use dots for individual data points for clarity
    plt.scatter(np.arange(1, N+1), time_series, color='black', s=40, label='Data Points', zorder=3)
    
    plt.xlabel('Time (Sample Index)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=12, fontweight='bold')  # Add y-axis label for completeness
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if FD is not None:
        plt.title(f'Fractal Time Series with $FD_{{th}}={FD}$', fontsize=12, fontweight='bold')
    else:
        plt.title('Fractal Time Series', fontsize=16)
    
    plt.legend(fontsize=16, prop={'weight': 'bold'})
    
    # Save the figure as a high-quality vector graphic for publication
    #plt.savefig('fractal_time_series.svg', format='svg', bbox_inches='tight')
    plt.show()

from itertools import cycle    
import matplotlib.pyplot as plt
import numpy as np

def illustrate_higuchi_parameters(time_series, cases):
    N = len(time_series)
    fig, axs = plt.subplots(len(cases), 1, figsize=(6, 5), dpi=200)
    
    if len(cases) == 1:
        axs = [axs]
    else:
        axs = axs.ravel()

    color_palette = [
        '#2ca02c',  # Green
        '#ff7f0e',  # Orange
        '#1f77b4',  # Blue
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Lime Green
        '#17becf',  # Cyan
        '#8c564b'   # Brown
    ]
    colors = cycle(color_palette)

    for idx, (m, k) in enumerate(cases):
        ax = axs[idx]
        current_color = color_palette[m - 1]
        
        if idx == 0:
            ax.plot(np.arange(1, N + 1), time_series, color='black', lw=1.5, label='Original')
        else:
            ax.plot(np.arange(1, N + 1), time_series, color='black', lw=1.5)

        ax.scatter(np.arange(1, N + 1), time_series, color='grey', s=20, alpha=0.7)
        ax.plot(np.arange(m, N + 1, k), time_series[m - 1::k], 'o-', color=current_color, lw=2, markersize=6, 
                markeredgecolor='black', label=f'Segment (m={m}, k={k})')

        for i, idxx in enumerate(range(m, N + 1, k)):
            text_offset = (0, 12) if i % 2 == 0 else (0, -16)
            label = f'X({m}+{i}k)' if i != 0 else f'X({m})'
            ax.annotate(label, (idxx, time_series[idxx - 1]), textcoords="offset points", xytext=text_offset, 
                        ha='center', fontsize=10, fontweight='bold', color='black')
        
        # Make x-ticks and y-ticks bold
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.set_tick_params(labelsize=10, labelrotation=0, labelcolor='black', grid_color='gray', grid_alpha=0.7)
        ax.yaxis.set_tick_params(labelsize=10, labelrotation=0, labelcolor='black', grid_color='gray', grid_alpha=0.7)
        plt.setp(ax.get_xticklabels(), fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontweight='bold')

        ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
        
        if idx == len(cases) - 1:
            ax.set_xlabel('Time (Sample Index)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # Keep x-ticks only for the last subplot
        else:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-ticks for other subplots

        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend inside each subplot in the bottom-right corner
        ax.legend(loc='lower right', fontsize=10, prop={'weight': 'bold'})

    # Adjusting the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0.1)

    plt.show()

    
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np


def illustrate_higuchi_Lmk_values(time_series, k_list):
    fig = plt.figure(figsize=(10, 2 * len(k_list)),dpi=100)
    gs = gridspec.GridSpec(len(k_list) + sum(np.diff(k_list) > 2), 1, figure=fig)
    
    current_gs_idx = 0  # Track the current GridSpec index
    color_palette = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']  # Specific colors for k=3
    default_color = '#708090'  # Scientifically suitable color (slate gray)
    
    for idx, k in enumerate(k_list):
        m_values = range(1, k + 1)
        L_values = [Lmk(time_series, m, k) for m in m_values]  # Calculate Lmk values

        if idx > 0 and k_list[idx] - k_list[idx-1] > 2:
            ax_dots = fig.add_subplot(gs[current_gs_idx, 0])
            ax_dots.text(0.5, 0.5, '•\n•\n•', fontsize=30, va='center', ha='center', linespacing=0.7)
            ax_dots.axis('off')
            current_gs_idx += 1
        
        ax = fig.add_subplot(gs[current_gs_idx, 0])
        current_gs_idx += 1

        L_sum = sum(L_values) if L_values else 0
        Lk = L_sum / k if L_values else 0
        
        if k == 3:  # Apply specific colors for k=3
            for m, L_value in zip(m_values, L_values):
                ax.bar(m, L_value, color=color_palette[m-1], alpha=0.7)
        else:  # Use the default color for all bars
            for m, L_value in zip(m_values, L_values):
                ax.bar(m, L_value, color=default_color, alpha=0.7)
        
        ax.set_title(f'$L_m(k)$ values for $k$={k}',fontsize=16, fontweight='bold')
        ax.set_xlabel('$m$',fontsize=1, fontweight='bold')
        ax.set_ylabel('$L_m(k)$',fontsize=16, fontweight='bold')
        ax.set_xticks(m_values)  # Set tick positions
        ax.tick_params(axis='both', which='major', labelsize=16)  # Set font size for ticks
        plt.setp(ax.get_xticklabels(), fontweight='bold')  # Make x-tick labels bold
        plt.setp(ax.get_yticklabels(), fontweight='bold')  # Make y-tick labels bold
        ax.tick_params(axis='x', labelsize=16, labelrotation=0)  # Style the tick labels

        
        # Create custom legend entries
        l_sum_patch = mpatches.Patch(color='none', label=f'$\Sigma L_m(k)$ = {L_sum:.2f}')
        lk_patch = mpatches.Patch(color='none', label=f'$L(k)$ = {Lk:.2f}')
        
        # Place the legend outside the plot on the right side
        ax.legend(handles=[l_sum_patch, lk_patch], loc='center left', bbox_to_anchor=(1, 0.5), prop={'weight':'bold','size':16}, frameon=True)
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.show()

    
#%% ----- Higuchi Parameters - Part III:  L_(k), FD ----- 


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Ensure your mean_Lk and compute_FD functions are defined here

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Ensure your mean_Lk and compute_FD functions are defined here

def plot_L_and_FD(time_series, k_list):
    num_plots = len(k_list)
    FD_values = []

    if num_plots == 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=250)  # Only 2 subplots for single k value
    else:
        fig, axs = plt.subplots(num_plots, 3, figsize=(18, 6 * num_plots), dpi=200)

    # Variables to store min and max values for the axes
    min_log_inv_k = float('inf')
    max_log_inv_k = float('-inf')
    min_log_L = float('inf')
    max_log_L = float('-inf')

    for idx, max_k in enumerate(k_list):
        k_values = list(range(1, int(max_k) + 1))
        L_values = [mean_Lk(time_series, k) for k in k_values]

        # Compute log(1/k) values for the new plot
        log_inv_k_values = [np.log10(1 / k) for k in k_values]
        valid_L_values = [l for l in L_values if l > 0]
        log_L_values = [np.log10(l) for l in valid_L_values]

        # Update min and max values for axes
        min_log_inv_k = min(min_log_inv_k, min(log_inv_k_values))
        max_log_inv_k = max(max_log_inv_k, max(log_inv_k_values))
        min_log_L = min(min_log_L, min(log_L_values))
        max_log_L = max(max_log_L, max(log_L_values))

        # Calculate the slope for log-log plot
        FD = compute_FD(time_series, max_k)
        slope = FD
        FD_values.append(FD)

        if num_plots == 1:
            # Plot L(k) values
            axs[0].plot(k_values, L_values, 'o-', color='#3498db', label='$L(k)$')
            axs[0].set_title(r'Mean curve length $L(k)$ values', fontsize=12, fontweight='bold')
            axs[0].set_xlabel('k values', fontsize=16, fontweight='bold')
            axs[0].set_ylabel('$L(k)$', fontsize=16, fontweight='bold')
            axs[0].grid(alpha=0.2)
            axs[0].legend(fontsize=16, loc='best', prop={'weight': 'bold'})

            # Plot log-log values
            axs[1].plot(log_inv_k_values, log_L_values, 'o-', color='purple', label=f'Log-Log plot (Slope = {slope:.2f})')
            axs[1].set_title(r'Log-Log plot', fontsize=12, fontweight='bold')
            axs[1].set_xlabel(r'$\log(1/k)$', fontsize=16, fontweight='bold')
            axs[1].set_ylabel('Log($L(k)$)', fontsize=16, fontweight='bold')
            axs[1].grid(alpha=0.2)
            axs[1].legend(fontsize=16, loc='best', prop={'weight': 'bold'})
        else:
            # Plot L(k) values
            axs[idx, 0].plot(k_values, L_values, 'o-', color='#3498db', label='$L(k)$')
            axs[idx, 0].set_title(r'Mean curve length $L(k)$ values for $K_{\max}$ = ' + str(max_k), fontsize=12, fontweight='bold')
            axs[idx, 0].set_ylabel('$L(k)$', fontsize=16, fontweight='bold')
            axs[idx, 0].set_xticks(k_values)
            axs[idx, 0].grid(alpha=0.2)
            axs[idx, 0].legend(fontsize=16, loc='best', prop={'weight': 'bold'})

            # Plot log_L_values vs log_inv_k_values with slope in legend
            axs[idx, 1].plot(log_inv_k_values, log_L_values, 'o-', color='purple', label=f'Log-Log plot (Slope = {slope:.2f})')
            axs[idx, 1].set_title(r'Log-Log plot for $K_{\max}$ = ' + str(max_k), fontsize=12, fontweight='bold')
            axs[idx, 1].set_xlabel(r'$\log(1/k)$', fontsize=16, fontweight='bold')
            axs[idx, 1].set_ylabel('Log($L(k)$)', fontsize=16, fontweight='bold')
            axs[idx, 1].grid(alpha=0.2)
            axs[idx, 1].legend(fontsize=16, loc='best', prop={'weight': 'bold'})

            # Plotting FD value
            axs[idx, 2].axhline(FD, color='#e74c3c', label=f'FD = {FD:.2f}')
            axs[idx, 2].set_title(r'Fractal Dimension (FD) for $K_{\max}$ = ' + str(max_k), fontsize=12, fontweight='bold')
            axs[idx, 2].set_ylabel('FD', fontsize=16, fontweight='bold')
            axs[idx, 2].set_xticks(k_values)
            axs[idx, 2].grid(alpha=0.2)
            axs[idx, 2].legend(fontsize=16, loc='best', prop={'weight': 'bold'})

            if idx == num_plots - 1:
                axs[idx, 0].set_xlabel('k values', fontsize=16, fontweight='bold')
                axs[idx, 1].set_xlabel(r'$\log(1/k)$', fontsize=16, fontweight='bold')
                axs[idx, 2].set_xlabel('k values', fontsize=16, fontweight='bold')

    # Now set the same axis limits for all subplots in the second column
    if num_plots > 1:
        for idx in range(num_plots):
            axs[idx, 1].set_xlim(min_log_inv_k, max_log_inv_k)
            axs[idx, 1].set_ylim(min_log_L, max_log_L)

    # Make ticks bold for all subplots
    if num_plots == 1:
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust font size
            plt.setp(ax.get_xticklabels(), fontweight='bold')  # Make x-ticks bold
            plt.setp(ax.get_yticklabels(), fontweight='bold')  # Make y-ticks bold
    else:
        for ax_row in axs:
            for ax in ax_row:
                ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust font size
                plt.setp(ax.get_xticklabels(), fontweight='bold')  # Make x-ticks bold
                plt.setp(ax.get_yticklabels(), fontweight='bold')  # Make y-ticks bold

    plt.tight_layout()
    plt.show()


import matplotlib.ticker as ticker
def plot_FD_vs_Kmax(Kmax_list, time_series):
    # Calculate FD for each K_max
    FD_values = [compute_FD(time_series, k) for k in Kmax_list]
    # Highlighting a specific K_max value as an example
    highlight_k = 5
    highlight_fd = FD_values[highlight_k - 1]
    
    # Plotting FD vs K_max
    fig, ax = plt.subplots(figsize=(10, 6),dpi=250)
    ax.plot(Kmax_list, FD_values, '-o', color='#2b8cbe', markerfacecolor='#a6bddb', markeredgewidth=2, markersize=10, linewidth=2, label=f'$FD_{{th}}={highlight_fd}$')
    
    #ax.set_title('FD vs $K_{\max}$', fontsize=16, fontweight='bold')
    ax.set_xlabel('K', fontsize=16, fontweight='bold')
    ax.set_ylabel('HFD', fontsize=16, fontweight='bold')
    ax.set_ylim([1, 2])
    ax.grid(True, which='both', linestyle='--', alpha=0.3)#, color='gray')
    #ax.set_facecolor('#f5f5f5')  # Adding a subtle background color
    ax.legend(fontsize=18, frameon=True, shadow=True, fancybox=True, prop={'weight': 'bold'})
    
    # Customizing tick labels
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=14, direction='out', length=6, width=2, colors='#333333')
    plt.setp(ax.get_xticklabels(), fontweight='bold')  # Make x-tick labels bold
    plt.setp(ax.get_yticklabels(), fontweight='bold')  # Make y-tick labels bold
    
    #Annotate
    ax.annotate(f'$K={highlight_k}$\n$HFD={highlight_fd:.2f}$', xy=(highlight_k, highlight_fd),
                xytext=(highlight_k + 2, highlight_fd + 0.05), textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
                fontsize=14, fontweight='bold',  # Make annotation bold
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='#2b8cbe', facecolor='#e6eff7'))

    plt.tight_layout()  # Adjust the padding between and around subplots
    plt.show()
    return FD_values


    
def plot_higuchi_fd(time_series_generator, series_type, FD_values, frequency_list, lamda=None):
    fig, axs = plt.subplots(len(frequency_list), len(FD_values), figsize=(15, 20))
    
    for fd_index, FD in enumerate(FD_values):
        H = 2 - FD  # Convert fractal dimension to Hurst exponent

        for freq_index, freq in enumerate(frequency_list):
            t = np.linspace(0, 1 - 1 / freq, freq)
            N = len(t)
            K_max = int((N / 2))
            Kmax_list = np.linspace(2, K_max, K_max, dtype=int)

            if series_type == 'wcf' and lamda is not None:
                time_series = time_series_generator(freq, lamda, K_max, H)
            else:
                time_series = time_series_generator(N, H)

            FD_values_series = [compute_FD(time_series, k) for k in Kmax_list]

            ax = axs[freq_index, fd_index]
            ax.plot(FD_values_series, label=f'Freq {freq}')
            ax.axhline(y=FD, color='r', linestyle='--')
            ax.set_ylim(1, 2.2)
            if freq_index == 0:
                ax.set_title(f'Higuchi FD calculation for {series_type} with FD = {FD}')
            if freq_index == len(frequency_list) - 1:
                ax.set_xlabel('Kmax')
            if fd_index == 0:
                ax.set_ylabel('FD')
            ax.legend()

    fig.tight_layout()
    plt.show()
    