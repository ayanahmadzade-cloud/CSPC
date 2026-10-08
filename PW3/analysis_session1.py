import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

cols = ["age", "chol", "trestbps", "thalach"]

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
axes = axes.flatten()

for i, c in enumerate(cols):
    axes[i].hist(df[c], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(c)

plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

print("--- Normality Check ---")
for c in cols:
    stat, p = stats.shapiro(df[c])
    res = "normal" if p > 0.05 else "not normal"
    print(f"{c}: p-val = {p:.5f} ({res})")