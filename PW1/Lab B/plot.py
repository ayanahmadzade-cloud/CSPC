import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

# 1. Read data from CSV (skip header line)
data = np.loadtxt('decay_observed.csv', delimiter=',', skiprows=1)
t = data[:, 0]
observed = data[:, 1]

# 2. Define N0 as the first observed count and calculate analytical law
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# 3. Create 1x2 subplot with shared x and y axes
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

# Left plot: Observed data (scatter)
ax1.scatter(t, observed, color='blue', label='Observed Data', s=15)
ax1.set_title("Observed Decay")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Count")
ax1.grid(True)

# Right plot: Analytical decay law (line)
ax2.plot(t, analytical, color='red', label='Analytical Law')
ax2.set_title("Analytical Decay Law")
ax2.set_xlabel("Time (s)")
ax2.grid(True)

plt.tight_layout()

# 4. Save output figure
plt.savefig("figure.png")
print("figure.png successfully generated!")