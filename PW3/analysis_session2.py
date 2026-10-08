import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# session 2 - heart disease part
df_heart = pd.read_csv("heart.csv")

h = df_heart[df_heart['target'] == 0]['thalach']
d = df_heart[df_heart['target'] == 1]['thalach']

t_stat, p_val = stats.ttest_ind(h, d, equal_var=False)
print(f"t-test result: t = {t_stat:.3f}, p = {p_val:.5e}")

plt.figure(figsize=(6, 4))
plt.bar(['Healthy', 'Disease'], [h.mean(), d.mean()], yerr=[h.sem(), d.sem()], capsize=5, color=['green', 'red'])
plt.ylabel('mean thalach')
plt.title('thalach by target')
plt.savefig("thalach_comparison.png")
plt.close()

# correlation age vs thalach
r, p = stats.pearsonr(df_heart['age'], df_heart['thalach'])
print(f"age vs thalach correlation: r = {r:.3f}, p = {p:.5e}")

plt.figure(figsize=(6, 4))
plt.scatter(df_heart['age'], df_heart['thalach'], alpha=0.5)
plt.xlabel('age')
plt.ylabel('thalach')
plt.savefig("age_vs_thalach.png")
plt.close()


# session 2 - chemicals part
df_chem = pd.read_csv("chemicals_cancer.csv")

# naive correlation
print("\n--- Naive Correlation ---")
print("benzene vs malignancy:", stats.pearsonr(df_chem['benzene'], df_chem['malignancy'])[0])
print("cadmium vs malignancy:", stats.pearsonr(df_chem['cadmium'], df_chem['malignancy'])[0])

# controlling pollution index
p_med = df_chem['pollution_index'].median()
df_ctrl = df_chem[(df_chem['pollution_index'] >= p_med - 10) & (df_chem['pollution_index'] <= p_med + 10)]

print("\n--- Controlled Correlation (Pollution ~ median) ---")
print("benzene vs malignancy:", stats.pearsonr(df_ctrl['benzene'], df_ctrl['malignancy'])[0])
print("cadmium vs malignancy:", stats.pearsonr(df_ctrl['cadmium'], df_ctrl['malignancy'])[0])