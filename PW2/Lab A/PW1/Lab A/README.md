## PW1 --- Lab A

- **pytest status**: 3/3 tests passed.
- **Speed comparison (200,000 atoms)**:
  - Pure-Python loop: 2.8507 s
  - NumPy vectorised: 0.0003 s
  - Speed-up factor: NumPy is ~9955x faster.
- **Conclusion**: Vectorised operations using NumPy significantly outperform standard Python loops for large particle simulations by eliminating Python loop overhead.