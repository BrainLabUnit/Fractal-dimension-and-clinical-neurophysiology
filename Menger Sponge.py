#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:13:38 2024

@author: sadafmoaveninejad
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_menger_sponge(level, size=3):
    if level == 0:
        return np.ones((size, size, size))
    
    size //= 3
    small_sponge = generate_menger_sponge(level-1, size)
    sponge = np.zeros((3*size, 3*size, 3*size))
    
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if (x != 1 or y != 1) and (y != 1 or z != 1) and (z != 1 or x != 1):
                    sponge[x*size:(x+1)*size, y*size:(y+1)*size, z*size:(z+1)*size] = small_sponge
                    
    return sponge

def plot_menger_sponge(sponge, facecolor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    
    ax.voxels(sponge, facecolors=facecolor, edgecolor='none')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()

level = 4
size = 3 ** level
sponge = generate_menger_sponge(level, size)

# Set colors for the entire sponge
facecolor = 'thistle'  # Change this to 'lightturquoise' if preferred

plot_menger_sponge(sponge, facecolor)