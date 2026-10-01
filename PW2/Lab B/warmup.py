import numpy as np
from scipy.optimize import minimize, newton

# 2A. Easy Convex Function: f(x) = (x - 3)^2 + 1
def f(x):
    return (x - 3)**2 + 1

def df(x):
    return 2 * (x - 3)

def d2f(x):
    return 2

print("=== Part 2A: Easy Convex Function ===")
# 1. Gradient Descent
x_gd = 0.0
lr = 0.1
for _ in range(100):
    x_gd = x_gd - lr * df(x_gd)
print(f"Gradient descent result: x = {x_gd:.4f}")

# 2. Newton's method
x_newton = newton(df, x0=0.0, fprime=d2f)
print(f"Newton's method result: x = {x_newton:.4f}")

# 3. SLSQP
res_slsqp = minimize(f, x0=[0.0], method="SLSQP")
print(f"SLSQP result: x = {res_slsqp.x[0]:.4f}\n")


# 2B. Harder Landscape: g(x) = x^4 - 3*x^2 + x + 5
def g(x):
    return x**4 - 3*x**2 + x + 5

def dg(x):
    return 4*x**3 - 6*x + 1

def d2g(x):
    return 12*x**2 - 6

def test_harder_landscape(x0_start):
    print(f"--- Running 2B with x0 = {x0_start} ---")
    x_gd = float(x0_start)
    lr = 0.01
    for _ in range(1000):
        x_gd = x_gd - lr * dg(x_gd)
    print(f"Gradient descent result: x = {x_gd:.4f}")

    try:
        x_newt = newton(dg, x0=x0_start, fprime=d2g)
        curvature = d2g(x_newt)
        point_type = "Minimum" if curvature > 0 else ("Maximum" if curvature < 0 else "Stationary")
        print(f"Newton result: x = {x_newt:.4f} (g''(x) = {curvature:.2f} -> {point_type})")
    except Exception as e:
        print(f"Newton method failed: {e}")

    res = minimize(g, x0=[x0_start], method="SLSQP")
    print(f"SLSQP result: x = {res.x[0]:.4f}\n")

print("=== Part 2B: Harder Landscape ===")
test_harder_landscape(0.0)
test_harder_landscape(2.0)