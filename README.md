## PW1 --- Lab A

- **pytest status**: 3/3 tests passed.
- **Speed comparison (200,000 atoms)**:
  - Pure-Python loop: 2.8507 s
  - NumPy vectorised: 0.0003 s
  - Speed-up factor: NumPy is ~9955x faster.
- **Conclusion**: Vectorised operations using NumPy significantly outperform standard Python loops for large particle simulations by eliminating Python loop overhead.


## PW1 Lab B

- **Data Results:** The observed radioactive decay data matches the theoretical analytical law ($N_0 e^{-\lambda t}$) very closely.
- **Snakemake Pipeline:** The Snakemake pipeline automates figure generation by monitoring input files and rebuilding `figure.png` only when `decay_observed.csv` or `plot.py` is updated.


## PW2 Lab A - Motion from Tracking Data

- **Measured Mean Acceleration:** -8.58 m/s²
- **Acceleration Standard Deviation:** 28.72 m/s²
- **Noise Observation:** Differentiation amplifies measurement noise because it calculates local differences between adjacent points, causing individual acceleration values to fluctuate significantly despite smooth position data.
- **Back Integration Results:** Re-integrating the acceleration back to position recovered the original trajectory with a maximum difference of 0.78 m, demonstrating that integration suppresses noise.
- **Bonus Task:** Completed 2D trajectory path analysis and speed calculation over time.


# Lab B: Optimization in Chemistry - Results Summary

## Part 2: Optimization Warmup
- Successfully tested Gradient Descent, Newton, and SLSQP methods on standard test functions.

## Part 3: Kinetics
- Reaction rate constant (k): Calculated using SLSQP minimization.
- Outputs generated: `kinetics.csv` and `kinetics.png`.

## Part 4: Chemical Equilibrium
- Reaction extent at equilibrium (x): Solved via `root_scalar` (brentq) and SLSQP.
- Outputs generated: `equilibrium.png`.

## Part 5: Titration Curve
- Equivalence point volume: 25.0 mL
- Calculated pH curve using charge balance and `root_scalar`.
- Outputs generated: `titration.png`.

## Part 6: Automation Workflow
- Snakemake workflow configured in `Snakefile` to automate the execution of all scripts and figure generation.