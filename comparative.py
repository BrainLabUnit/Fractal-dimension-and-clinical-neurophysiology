#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:45:41 2024

@author: sadafmoaveninejad
"""

import numpy as np
import matplotlib.pyplot as plt
from functions_comparative import generate_wcf, generate_tlf, generate_fbm, generate_fgn
from functions_comparative import compute_fractal_dimension

#%%

# Define theoretical FD values
fd_theoretical = np.arange(1.05, 2, 0.1)
he_theoretical = 2-fd_theoretical
fs = 512
t = 1
N = fs*t
std_dev = 10

# Initialize results storage
results = {
    'sPSD': [],
    'DFA': [],
    'HE': [],#'GHE'
    'HFD': [],
    'KFD': []#,
    #'BC': []
}

# For each FDTH value, generate the synthetic time series and compute fractal dimensions
for fd in fd_theoretical:
    print('Theoretical FD', fd)
    # Generate each synthetic time series with the given FDTH
    wcf = generate_wcf(fd, N)
    tlf = generate_tlf(fd, fs)
    fbm = generate_fbm(fd, N, std_dev)
    fgn = generate_fgn(fd,N)
    
    # Compute fractal dimensions using each method
    for method in results.keys():
        print(method,'----')
        
        if method == 'DFA':
            fd_estimated_wcf = 3 - compute_fractal_dimension(method, wcf,fs)
            fd_estimated_tlf = 3 - compute_fractal_dimension(method, tlf,fs)
            fd_estimated_fbm = 3 - compute_fractal_dimension(method, fbm,fs)
            fd_estimated_fgn = 2 - compute_fractal_dimension(method, fgn,fs)
        elif method == 'sPSD':
            fd_estimated_wcf = (5 - compute_fractal_dimension(method, wcf,fs))/2
            fd_estimated_tlf = (5 - compute_fractal_dimension(method, tlf,fs))/2
            fd_estimated_fbm = (5 - compute_fractal_dimension(method, fbm,fs))/2
            fd_estimated_fgn = (3 - compute_fractal_dimension(method, fgn,fs))/2
        else:
            
            fd_estimated_wcf = compute_fractal_dimension(method, wcf,fs)
            fd_estimated_tlf = compute_fractal_dimension(method, tlf,fs)
            fd_estimated_fbm = compute_fractal_dimension(method, fbm,fs)
            fd_estimated_fgn = compute_fractal_dimension(method, fgn,fs)
        
        # Store the results
        results[method].append({
            'WCF': fd_estimated_wcf,
            'TLF': fd_estimated_tlf,
            'fBm': fd_estimated_fbm,
            'fGn': fd_estimated_fgn
            
        })

#---------------------------- Plotting ---------------------------------
# Plotting the comparative analysis with adjustments
fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=150, constrained_layout=True)

# Define markers for different curves
markers = ['o', '^', 's', 'D']  # Example markers: circle, triangle_up, square, diamond

# Create lines for the legend outside the loop, using plt.plot([], [], ...) for an empty plot as legend proxy
lines = [
    plt.plot([], [], marker=markers[0], linestyle='-', color='blue', label='WCF')[0],
    plt.plot([], [], marker=markers[1], linestyle='-', color='orange', label='TLF')[0],
    plt.plot([], [], marker=markers[2], linestyle='-', color='green', label='fBm')[0],
    plt.plot([], [], marker=markers[3], linestyle='-', color='red', label='fGn')[0],
    plt.plot([], [], linestyle='--', color='black', label='Ideal FD=FDth')[0]#FD=FDth
]

for i, method in enumerate(results.keys()):
    ax = axs.flatten()[i]
    
    # Add light gray background for specific region    
    ax.fill_betweenx([1, 2], 1, 2, color='lightgray', alpha=0.5)

    # Prepare data for plotting
    fd_estimates_wcf = [result['WCF'] for result in results[method]]
    fd_estimates_tlf = [result['TLF'] for result in results[method]]
    fd_estimates_fbm = [result['fBm'] for result in results[method]]
    fd_estimates_fgn = [result['fGn'] for result in results[method]]
    
    # Plotting the results for each STS with different markers
    ax.plot(fd_theoretical, fd_estimates_wcf, marker=markers[0], linestyle='-', color='blue', label='WCF')
    ax.plot(fd_theoretical, fd_estimates_tlf, marker=markers[1], linestyle='-', color='orange', label='TLF')
    ax.plot(fd_theoretical, fd_estimates_fbm, marker=markers[2], linestyle='-', color='green', label='fBm')
    ax.plot(fd_theoretical, fd_estimates_fgn, marker=markers[3], linestyle='-', color='red', label='fGn')
        
    
    # Add title with bold text
    ax.set_title(f'{method} Estimates', fontsize=14, fontweight='bold')
    
    # Add ideal line
    ax.plot(fd_theoretical, fd_theoretical, 'k--', label='Ideal FD=FDth')  # Ideal line does not need a marker
        
    # Add axis labels with bold text
    ax.set_xlabel('Theoretical FD', fontsize=12, fontweight='bold')
    #ax.set_ylabel('Estimated HE', fontsize=12, fontweight='bold')
        
    if i % 3 == 0:  # Only for subplots on the left
    # ax.set_ylabel('Estimated FD', fontsize=12, fontweight='bold')
        ax.set_ylabel('Estimated FD', fontsize=12, fontweight='bold')
        
    # Make grid lines and axis limits consistent
    ax.grid(True)
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 2.5)
    
    # Customize tick labels and make them bold
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

# Place the legend in the subplot reserved for the 6th plot if there are fewer methods
if len(results) < 6:
    # Making the legend box bigger with bold font
    axs.flatten()[-1].legend(handles=lines, loc='center', fontsize=12, prop={'weight': 'bold'}, frameon=True, shadow=True, fancybox=True)
    axs.flatten()[-1].axis('off')  # Hide axes for the legend subplot

plt.show()



#%%  MSE
# Assuming the 'results' dictionary is already filled with FD estimates as in your provided code
import pandas as pd

# Function to calculate MSE
def calculate_mse(estimated, ideal):
    mse = np.mean((np.array(estimated) - np.array(ideal))**2)
    return mse

# Extract the FD values for each method and time series
fd_values = {ts: {method: [] for method in results.keys()} for ts in ['WCF', 'TLF', 'fBm', 'fGn']}

for method, estimates in results.items():
    for est in estimates:
        for ts in ['WCF', 'TLF', 'fBm', 'fGn']:
            fd_values[ts][method].append(est[ts])

# Calculate MSE for each time series and method
mse_table = {ts: {method: calculate_mse(fd_values[ts][method], fd_theoretical) for method in results.keys()} for ts in ['WCF', 'TLF', 'fBm', 'fGn']}

# Convert the MSE results to a DataFrame for easy display
df_mse = pd.DataFrame(mse_table).T  # Transpose to have methods as columns and time series as rows

print(df_mse)

