import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize

# Equilibrium constant
K = 50.0

# Define k_imbalance function
def k_imbalance(x):
    return ((2 * x)**2) / ((1 - x) * (1 - x)) - K

# 1. Safe root-finding (using brentq within bounds 0 and 1)
sol = root_scalar(k_imbalance, bracket=[0.001, 0.999], method='brentq')
x_newton = sol.root

# 2. SLSQP minimization on k_imbalance(x)^2
def objective(x):
    return k_imbalance(x[0])**2

res = minimize(objective, x0=[0.5], method="SLSQP", bounds=[(0.001, 0.999)])
x_slsqp = res.x[0]

print(f"Equilibrium x (Root-finding): {x_newton:.4f}")
print(f"Equilibrium x (SLSQP):        {x_slsqp:.4f}")

# Equilibrium amounts
nH2_eq = 1 - x_newton
nI2_eq = 1 - x_newton
nHI_eq = 2 * x_newton

print(f"Equilibrium amounts: H2 = {nH2_eq:.3f} mol, I2 = {nI2_eq:.3f} mol, HI = {nHI_eq:.3f} mol")

# Plotting composition changes
x_vals = np.linspace(0.01, 0.95, 200)
nH2 = 1 - x_vals
nI2 = 1 - x_vals
nHI = 2 * x_vals

plt.figure(figsize=(8, 5))
plt.plot(x_vals, nH2, label="H2", color="blue")
plt.plot(x_vals, nI2, label="I2", color="green", linestyle="--")
plt.plot(x_vals, nHI, label="HI", color="red")
plt.axvline(x=x_newton, color="black", linestyle=":", label=f"Equilibrium x ≈ {x_newton:.2f}")

plt.xlabel("Reaction Extent (x)")
plt.ylabel("Amount (moles)")
plt.title("Chemical Equilibrium Composition")
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png")
print("equilibrium.png saved successfully!")