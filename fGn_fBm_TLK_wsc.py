#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:02:15 2024

@author: sadafmoaveninejad
"""

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

#%% Functions

# Takagi–Landsberg function (blancmange curve)
def blancmange(FD, fs, x=None, step=0.005):
    H = 2 - FD  # Calculate H based on the provided fractal dimension
    w = 2 ** (-H)  # Compute w
    if x is None:
        x = 1 / fs  # Compute x based on the sampling frequency if not provided
    n_final = 100  # Default maximum summation limit
    print(x,step)
    # Initialize arrays to store values
    x_values = np.arange(0, x + step, step)
    B = np.zeros_like(x_values)  # Blancmange values

    # Compute Blancmange-Takagi function
    for R, x1 in enumerate(x_values):
        org = np.array([2 ** n * x1 for n in range(n_final + 1)])
        a1 = w ** np.arange(n_final + 1) * np.abs(org - np.round(org))
        B[R] = np.sum(a1)
    return B[:-1]

# Weierstrass cosine function
def wsc(N=1000, lamda=5, M=26, H=0.1):
    if lamda < 1:
        raise ValueError('lamda must be greater than 1.')
    if H < 0 or H > 1:
        raise ValueError('H must be between 0 and 1.')
    
    # Ensure lamda is a float to avoid negative integer power error
    lamda = float(lamda)
    t = np.linspace(0, 1, N)
    xwsc = np.zeros(N)
    for i in range(N):
        # Calculate the powers separately to avoid integer exponentiation
        exponents = -np.arange(0, M+1) * H
        terms = np.power(lamda, exponents) * np.cos(2 * np.pi * np.power(lamda, np.arange(0, M+1)) * t[i])
        xwsc[i] = sum(terms)
    
    FDth = 2 - H
    print('Theoretical value of FD of time series:', FDth)
    print('Theoretical value of H of time series:', H)
    
    return xwsc

# Define the set_ax_style function
def set_ax_style(ax, title=None, x_label=None, y_label=None, ylim=None):
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=4)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10, fontweight='bold', labelpad=3)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10, fontweight='bold', labelpad=3)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    

#%%
# Define the parameters
fs = 256
T = 1  # Time in seconds
t = np.arange(0, T, 1/fs)
n_points = len(t)

landa = 5
K = 26
FD = 1.5
H = 2 - FD
w = 2 ** (-H)

mean, std_dev = 0, 1
fGn = np.random.normal(mean, std_dev * (1 / (2 ** H)), size=n_points)
# Generating a non-stationary signal (fractional Brownian motion) by cumulatively summing the fGn
fBm = np.cumsum(fGn)


# Generate WCF and TLF
wcf = wsc(n_points, landa, K, H)

n = 1               # (n) is the iteration number, in range (1 to 10).
w = 2 ** (-H)         # H_tak = -log2(w), D = log(4*w)/log(2)
tlf = blancmange(FD, fs,n, 1/fs)

# ---------------- Plotting --------------

# Adjust the figure and axes styling
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Define colors for each signal
colors = ['blue', 'green', 'red', 'purple']

# Plotting with distinct colors
set_ax_style(axs[0, 0], title='Fractional Gaussian Noise (fGn)',  y_label='Amplitude')
axs[0, 0].plot(t, fGn, color=colors[0])

set_ax_style(axs[0, 1], title='Fractional Brownian Motion (fBm)')
axs[0, 1].plot(t, fBm, color=colors[1])

set_ax_style(axs[1, 0], title='Weierstrass Cosine Function (WCF)', x_label='Time (s)', y_label='Amplitude')
axs[1, 0].plot(t, wcf, color=colors[2])

set_ax_style(axs[1, 1], title='Takagi-Landsberg Function (TLF)', x_label='Time (s)')
axs[1, 1].plot(t, tlf, color=colors[3])

# Adjust layout and show plot
plt.tight_layout()
plt.show()