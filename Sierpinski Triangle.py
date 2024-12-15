#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:12:38 2024

@author: sadafmoaveninejad
"""

#%% Sierpinski Triangle

import matplotlib.pyplot as plt
import numpy as np

def sierpinski_triangle(ax, vertices, depth, colors):
    if depth == 0:
        triangle = plt.Polygon(vertices, edgecolor='black', facecolor=colors[0])  # Base case with color
        ax.add_patch(triangle)
    else:
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]
        # Calculate the midpoints of each side
        m01 = (v0 + v1) / 2
        m12 = (v1 + v2) / 2
        m20 = (v2 + v0) / 2
        # Recursive calls for each sub-triangle, alternating colors
        sierpinski_triangle(ax, [v0, m01, m20], depth - 1, colors)
        sierpinski_triangle(ax, [v1, m01, m12], depth - 1, colors)
        sierpinski_triangle(ax, [v2, m12, m20], depth - 1, colors)
        # Draw the inverted triangle with a color based on the current depth
        inner_triangle_color = colors[depth % len(colors)]
        inner_triangle = plt.Polygon([m01, m12, m20], edgecolor='black', facecolor=inner_triangle_color)
        ax.add_patch(inner_triangle)

# Define the main triangle vertices
vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

# Depth of the recursion
depth = 7

# Define your colors here: "medium purple" and "lavender"
colors = ["lightsteelblue", "pink"]  # Alternating colors for different depths
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axis('off')
# Plot the Sierpinski Triangle with the modified color scheme
sierpinski_triangle(ax, vertices, depth, colors)
# Display the plot
plt.show()

