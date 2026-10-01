import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate synthetic reaction data directly
np.random.seed(42)
t_data = np.linspace(0, 10, 20)
C_true = 10.0 * np.exp(-0.25 * t_data)
C_data = C_true + np.random.normal(0, 0.2, size=len(t_data))

# Save kinetics.csv automatically
pd.DataFrame({"time": t_data, "concentration": C_data}).to_csv("kinetics.csv", index=False)

# Initial concentration C0
C0 = C_data[0]

# Total error function
def total_error(k):
    k_val = k[0]
    C_model = C0 * np.exp(-k_val * t_data)
    return np.sum((C_data - C_model)**2)

# Minimize using SLSQP
res = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]

print(f"Fitted rate constant k = {fitted_k:.4f}")

# Plotting
t_dense = np.linspace(t_data.min(), t_data.max(), 200)
C_dense = C0 * np.exp(-fitted_k * t_dense)

plt.figure(figsize=(8, 5))
plt.scatter(t_data, C_data, color="red", label="Measured Data")
plt.plot(t_dense, C_dense, color="blue", label=f"Fitted Curve (k ≈ {fitted_k:.2f})")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Reaction Rate Fitting (Kinetics)")
plt.legend()
plt.grid(True)
plt.savefig("kinetics.png")
print("Saved kinetics.png successfully!")