import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Sample 3D trajectory data
t = np.linspace(0, 10, 100)  # Time variable
x = np.sin(t)  # X-coordinates
y = np.cos(t)  # Y-coordinates
z = t  # Z-coordinates (height)

# Create 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory
ax.plot(x, y, z, label="3D Trajectory", color="b")

# Add labels
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("3D Trajectory Plot")

# Show legend
ax.legend()

# Show plot
plt.show()
