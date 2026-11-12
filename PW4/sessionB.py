import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ==========================================
# B1. LOAD AND HIDE LABELS
# ==========================================
print("=== B1. LOAD WINE DATA ===")
wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target

print(wine.DESCR[:700])
print("\nX.describe() summary:")
print(X.describe().round(2))

# ==========================================
# B2. PCA ON RAW DATA
# ==========================================
print("\n=== B2. PCA ON RAW DATA ===")
pca_raw = PCA()
pca_raw.fit(X)
print("Explained variance ratio (Raw PCA):", pca_raw.explained_variance_ratio_[:3].round(4))

pc1_raw_loadings = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)
print("\nPC1 Loadings (Raw data, sorted):\n", pc1_raw_loadings.round(4))

# ==========================================
# B3. SCALE, THEN PCA AGAIN
# ==========================================
print("\n=== B3. SCALE & PCA AGAIN ===")
scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca_scaled = PCA()
pca_scaled.fit(Z)
exp_var = pca_scaled.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("Explained variance ratio (Scaled):", exp_var[:5].round(4))
print("Cumulative variance (Scaled):", cum_var[:5].round(4))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.7, label="Individual")
ax.step(range(1, len(exp_var) + 1), cum_var, where="mid", label="Cumulative", color="darkorange")
ax.axhline(0.9, color="red", linestyle="--", label="90% Variance Threshold")
ax.set_xlabel("Principal Components")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Scree Plot (Scaled Wine Data)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/B3_scree_plot.png", dpi=150)
plt.close()

loadings_df = pd.DataFrame(
    pca_scaled.components_[:2].T, 
    index=X.columns, 
    columns=["PC1", "PC2"]
)
print("\nPC1 and PC2 Loadings (Scaled):\n", loadings_df.round(4))

Z_pca = pca_scaled.transform(Z)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.7, color="purple", edgecolor="k")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Wines projected on PC1-PC2 plane")
plt.tight_layout()
plt.savefig("figures/B3_pca_plane.png", dpi=150)
plt.close()

# ==========================================
# B4. K-MEANS
# ==========================================
print("\n=== B4. K-MEANS ===")
inertias = []
k_range = range(1, 11)
for k in k_range:
    km_sweep = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
    km_sweep.fit(Z)
    inertias.append(km_sweep.inertia_)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_range, inertias, marker="o", color="b")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("K-Means Elbow Method")
plt.tight_layout()
plt.savefig("figures/B4_elbow.png", dpi=150)
plt.close()

km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km.fit(Z)

# ==========================================
# B5. WHAT ARE THE CLUSTERS?
# ==========================================
print("\n=== B5. CLUSTER INTERPRETATION ===")
print("Cluster sizes:", np.bincount(km.labels_))

centers_real = scaler.inverse_transform(km.cluster_centers_)
centers_df = pd.DataFrame(centers_real, columns=X.columns)
print("\nCluster Centers in Real Units:\n", centers_df.T.round(2))

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km.labels_, cmap="viridis", alpha=0.8, edgecolor="k")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-Means Clusters on PC1-PC2 plane")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.tight_layout()
plt.savefig("figures/B5_clusters.png", dpi=150)
plt.close()

# ==========================================
# B6. THE REVEAL (AGREEMENT WITH CULTIVARS)
# ==========================================
print("\n=== B6. THE REVEAL ===")
def compute_agreement(labels, true_labels):
    ct = pd.crosstab(true_labels, labels)
    agreement = ct.max(axis=0).sum() / len(true_labels)
    return ct, agreement

ct_scaled, agr_scaled = compute_agreement(km.labels_, cultivar)
print("Crosstab (Scaled K-Means vs Cultivars):\n", ct_scaled)
print(f"Agreement (Scaled): {agr_scaled:.4f}")

km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(X)
_, agr_raw = compute_agreement(km_raw.labels_, cultivar)

km_pca = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(Z_pca[:, :2])
_, agr_pca = compute_agreement(km_pca.labels_, cultivar)

agreement_table = pd.DataFrame([
    {"Method": "Scaled Data", "Agreement": agr_scaled},
    {"Method": "Raw Data", "Agreement": agr_raw},
    {"Method": "PCA (PC1-PC2 only)", "Agreement": agr_pca}
])
print("\nAgreement Table:\n", agreement_table.round(4).to_string(index=False))
agreement_table.to_csv("results/B_agreement.csv", index=False)
print("Saved results/B_agreement.csv")