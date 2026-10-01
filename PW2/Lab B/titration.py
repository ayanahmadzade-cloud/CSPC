import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Titration parameters
V_acid = 25.0  # mL of acid
C_acid = 0.1   # M concentration of acid
C_base = 0.1   # M concentration of base
Kw = 1e-14

def titration_error(pH, V_b):
    """Calculates charge balance error for strong acid - strong base titration."""
    H = 10**(-pH)
    OH = Kw / H
    V_total = V_acid + V_b
    
    # Concentrations in the mixture
    c_A = (C_acid * V_acid) / V_total
    c_B = (C_base * V_b) / V_total
    
    # Charge balance equation: [H+] + [Na+] - [Cl-] - [OH-] = 0
    return H + c_B - c_A - OH

# Base volume range (0 to 50 mL)
V_base_list = np.linspace(0.1, 50.0, 200)
pH_list = []

for V_b in V_base_list:
    # Solve for pH using root-finding
    sol = root_scalar(titration_error, args=(V_b,), bracket=[0.0, 14.0], method='brentq')
    pH_list.append(sol.root)

# Find equivalence point (where V_base == V_acid)
v_eq = V_acid * C_acid / C_base
sol_eq = root_scalar(titration_error, args=(v_eq,), bracket=[0.0, 14.0], method='brentq')
pH_eq = sol_eq.root

print(f"Equivalence point volume: {v_eq:.2f} mL")
print(f"pH at equivalence point:  {pH_eq:.2f}")

# Plotting titration curve
plt.figure(figsize=(8, 5))
plt.plot(V_base_list, pH_list, color="purple", linewidth=2, label="Titration Curve")
plt.axvline(x=v_eq, color="gray", linestyle="--", label=f"Equivalence Point ({v_eq} mL)")
plt.axhline(y=pH_eq, color="gray", linestyle=":")

plt.xlabel("Volume of NaOH Added (mL)")
plt.ylabel("pH")
plt.title("Strong Acid - Strong Base Titration Curve")
plt.legend()
plt.grid(True)
plt.savefig("titration.png")
print("titration.png saved successfully!")