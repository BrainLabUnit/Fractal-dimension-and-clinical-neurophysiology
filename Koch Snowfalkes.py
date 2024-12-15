#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:11:30 2024

@author: sadafmoaveninejad
"""

# Koch Snowfalkes
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

def generate_koch_snowflake(order, scale=10):
    # Initial triangle points
    points = scale * np.array([[0, 0],
                               [1, 0],
                               [0.5, np.sin(np.pi / 3)],
                               [0, 0]])
    return koch_curve(points, order)

# Set the order of the Koch snowflake
order = 0

# Generate snowflake points
points = generate_koch_snowflake(order)

# Create the plot
fig, ax = plt.subplots()
ax.axis('equal')
ax.set_axis_off()

# Draw the colored lines
colors = ['red', 'green', 'blue']  # Define a list of colors
segment_length = len(points) // len(colors)
for i in range(len(points) - 1):
    plt.plot(points[i:i+2, 0], points[i:i+2, 1], color=colors[i // segment_length])

# Display the plot
plt.show()