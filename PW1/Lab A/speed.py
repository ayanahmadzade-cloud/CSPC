import time
from decay import simulate, simulate_loop

N0 = 200000
lam = 0.4

# Pure-Python loop version timing
t0 = time.perf_counter()
simulate_loop(N0, lam)
t1 = time.perf_counter()
loop_time = t1 - t0

# NumPy vectorised version timing
t2 = time.perf_counter()
simulate(N0, lam)
t3 = time.perf_counter()
numpy_time = t3 - t2

speedup = loop_time / numpy_time

print(f"Loop time: {loop_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy version is {speedup:.2f}x faster!")