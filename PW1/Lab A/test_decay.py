import numpy as np
import pytest
from decay import simulate, simulate_loop

def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000

# TODO 1: test_rejects_negative_rate
def test_rejects_negative_rate():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)

# TODO 2: test_matches_law
def test_matches_law():
    N0 = 10000
    lam = 0.4
    dt = 0.05
    steps = 200

    runs = [simulate(N0, lam, dt=dt, steps=steps, seed=s) for s in range(50)]
    avg_counts = np.mean(runs, axis=0)

    t = np.arange(steps + 1) * dt
    expected = N0 * np.exp(-lam * t)

    assert avg_counts == pytest.approx(expected, rel=1e-1)