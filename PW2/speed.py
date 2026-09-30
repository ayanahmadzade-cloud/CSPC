import numpy as np
import matplotlib.pyplot as plt

# Read trajectory data: time, x, y
data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

# Compute velocities using np.gradient
vx = np.gradient(x, t)
vy = np.gradient(y, t)

# Calculate speed: sqrt(vx^2 + vy^2)
speed = np.sqrt(vx**2 + vy**2)

# Plot 1: 2D Path (x vs y)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Trajectory', color='purple')
plt.title('2D Trajectory (x vs y)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('trajectory_path.png')
plt.close()

# Plot 2: Speed over time
plt.figure(figsize=(8, 5))
plt.plot(t, speed, label='Speed', color='green')
plt.title('Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('speed.png')
plt.close()

print("Bonus task completed successfully! Saved trajectory_path.png and speed.png.")