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