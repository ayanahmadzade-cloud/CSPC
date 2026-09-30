import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t, x, y = data[:, 0], data[:, 1], data[:, 2]

vx = np.gradient(x, t)
vy = np.gradient(y, t)
speed = np.sqrt(vx**2 + vy**2)

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

print("Bonus task completed successfully!")
