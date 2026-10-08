import numpy as np
import pandas as pd

np.random.seed(42)

# heart dataset
n = 300
age = np.random.randint(29, 78, n)
chol = np.random.normal(246, 50, n).astype(int)
trestbps = np.random.normal(131, 17, n).astype(int)
target = np.random.choice([0, 1], size=n, p=[0.45, 0.55])
thalach = np.where(target == 1, np.random.normal(139, 20, n), np.random.normal(158, 18, n)).astype(int)

df_heart = pd.DataFrame({
    'age': age,
    'sex': np.random.choice([0, 1], n),
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'target': target
})
df_heart.to_csv('heart.csv', index=False)

# chemicals cancer dataset
n2 = 1000
pollution = np.random.uniform(10, 90, n2)
cadmium = pollution * 0.8 + np.random.normal(0, 5, n2)
benzene = np.random.uniform(5, 50, n2)
age2 = np.random.randint(20, 80, n2)
malignancy = 10 + 0.7 * benzene + 0.8 * pollution + 0.1 * age2 + np.random.normal(0, 3, n2)

df_chem = pd.DataFrame({
    'benzene': np.round(benzene, 2),
    'cadmium': np.round(cadmium, 2),
    'pollution_index': np.round(pollution, 1),
    'age': age2,
    'malignancy': np.round(malignancy, 1)
})
df_chem.to_csv('chemicals_cancer.csv', index=False)
print("data files generated.")