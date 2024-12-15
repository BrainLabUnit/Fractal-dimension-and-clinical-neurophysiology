#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:40:00 2024

@author: sadafmoaveninejad
"""

# Koch Curve


import numpy as np
import matplotlib.pyplot as plt

def koch_curve(points, order):
    if order == 0:
        return points
    else:
        new_points = []
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            dist = np.linalg.norm(p1 - p0) / 3
            angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            p2 = p0 + np.array([np.cos(angle), np.sin(angle)]) * dist
            p3 = p2 + np.array([np.cos(angle - np.pi / 3), np.sin(angle - np.pi / 3)]) * dist
            p4 = p0 + np.array([np.cos(angle), np.sin(angle)]) * dist * 2
            new_points.extend([p0, p2, p3, p4])
        new_points.append(points[-1])
        return koch_curve(np.array(new_points), order - 1)

def generate_koch_curve(order, scale=10):
    points = scale * np.array([[1, 0], [0, 0]])
    return koch_curve(points, order)

# Set the order of the Koch curve
order = 1

# Generate Koch curve points
points = generate_koch_curve(order)

# Create the plot
fig, ax = plt.subplots()
ax.axis('equal')
ax.set_axis_off()

# Choose a fancy color
fancy_color = "slateblue"#"limegreen"  # darkturquoise
# fancy_color = "xkcd:azure"  # Unique shade of blue from the XKCD color survey

# Draw the Koch curve with the chosen fancy color
plt.plot(points[:, 0], points[:, 1], color=fancy_color)

# Display the plot
plt.show()
