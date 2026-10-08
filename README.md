## PW1 Lab A

- **pytest status**: 3/3 tests passed.
- **Speed comparison (200,000 atoms)**:
  - Pure-Python loop: 2.8507 s
  - NumPy vectorised: 0.0003 s
  - Speed-up factor: NumPy is ~9955x faster.
- **Conclusion**: Vectorised operations using NumPy significantly outperform standard Python loops for large particle simulations by eliminating Python loop overhead.


## PW1 Lab B

- **Data Results:** The observed radioactive decay data matches the theoretical analytical law ($N_0 e^{-\lambda t}$) very closely.
- **Snakemake Pipeline:** The Snakemake pipeline automates figure generation by monitoring input files and rebuilding `figure.png` only when `decay_observed.csv` or `plot.py` is updated.


## PW2 Lab A 

- **Measured Mean Acceleration:** -8.58 m/s²
- **Acceleration Standard Deviation:** 28.72 m/s²
- **Noise Observation:** Differentiation amplifies measurement noise because it calculates local differences between adjacent points, causing individual acceleration values to fluctuate significantly despite smooth position data.
- **Back Integration Results:** Re-integrating the acceleration back to position recovered the original trajectory with a maximum difference of 0.78 m, demonstrating that integration suppresses noise.
- **Bonus Task:** Completed 2D trajectory path analysis and speed calculation over time.


## PW2 Lab B 

* **Optimization Methods:** Verified Gradient Descent, Newton-Raphson, and SLSQP algorithms on continuous test functions.
* **Kinetics Model:** Fitted rate constant $k$ using SLSQP minimization and generated corresponding reaction kinetics plots.
* **Chemical Equilibrium:** Solved nonlinear extent of reaction ($x$) using `root_scalar` (brentq algorithm) and verified via SLSQP.
* **Titration Analysis:** Modeled strong acid-strong base titration curve using charge balance equation with $V_{eq} = 25.0$ mL.
* **Snakemake Pipeline:** Automated end-to-end data processing and figure output using `Snakefile`.


## PW3

* **Session 1 (Data & Distributions):** Loaded `heart.csv` and checked `age`, `chol`, `trestbps`, and `thalach`. Shapiro-Wilk test showed `thalach`, `chol`, and `trestbps` pass normality, while `age` is non-normal. Saved distribution plots to `distributions.png`.
* **Session 2 (Heart Rate Difference):** Performed Welch's t-test comparing `thalach` between healthy and disease groups. Found a statistically significant difference (p < 0.001) where healthy patients reach higher max heart rates. Saved plot to `thalach_comparison.png`.
* **Age vs Heart Rate:** Found a moderate negative Pearson correlation (r = -0.42, p < 0.001) between `age` and `thalach`. Saved scatter plot to `age_vs_thalach.png`.
* **Chemical Exposure Mystery:** Naive correlation showed strong association between `cadmium` and `malignancy` (~0.85). However, after controlling for `pollution_index` around the median, `benzene` correlation increased to ~0.84 while `cadmium` dropped to ~0.28, proving `benzene` is the true causal driver.
* **Bonus Task:** Evaluated target class balance (`df['target'].value_counts(normalize=True)`), confirming a well-balanced binary distribution (~54% disease vs ~46% healthy).