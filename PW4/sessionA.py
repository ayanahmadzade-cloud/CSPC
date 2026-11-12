import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ==========================================
# A2. CLEAN AND LOOK
# ==========================================
print("=== A2. CLEAN AND LOOK ===")
df = pd.read_csv("data/concrete.csv")
print("Initial shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values per column:\n", df.isna().sum())

df = df.drop_duplicates().reset_index(drop=True)
print("Shape after dropping duplicates:", df.shape)

X = df.drop(columns=["strength"])
y = df["strength"]

print("\nSummary statistics (df.describe()):")
print(df.describe().round(2))

# Plot strength vs cement and strength vs age
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df["cement"], df["strength"], alpha=0.4, color="steelblue")
axes[0].set_xlabel("Cement (kg/m³)")
axes[0].set_ylabel("Compressive Strength (MPa)")
axes[0].set_title("Strength vs Cement")

axes[1].scatter(df["age"], df["strength"], alpha=0.4, color="darkorange")
axes[1].set_xlabel("Age (days)")
axes[1].set_ylabel("Compressive Strength (MPa)")
axes[1].set_title("Strength vs Age")

plt.tight_layout()
plt.savefig("figures/A2_scatter.png", dpi=150)
plt.close()

# ==========================================
# A3. SPLIT
# ==========================================
print("\n=== A3. TRAIN / TEST SPLIT ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==========================================
# A4. LINEAR REGRESSION
# ==========================================
print("\n=== A4. LINEAR REGRESSION ===")
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train, y_train)

print(f"Intercept (beta_0): {lin_reg.intercept_:.4f}")
for col, coef in zip(X.columns, lin_reg.coef_):
    print(f"  {col:14s}: {coef:.6f}")

y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_adj_r2 = adj_r2(train_r2, len(X_train), X_train.shape[1])

lin_metrics = pd.DataFrame([{
    "Train RMSE": train_rmse,
    "Test RMSE": test_rmse,
    "Test MAE": test_mae,
    "Train R2": train_r2,
    "Train Adj R2": train_adj_r2,
    "Test R2": test_r2
}])
print("\nLinear Regression Metrics:")
print(lin_metrics.round(4).to_string(index=False))

# Residuals vs Age (log x-axis)
residuals = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test["age"], residuals, alpha=0.5, color="purple")
ax.axhline(0, color="black", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("Age (days, log scale)")
ax.set_ylabel("Residuals: True - Predicted (MPa)")
ax.set_title("Test Residuals vs Age (Log Scale)")
plt.tight_layout()
plt.savefig("figures/A4_residuals.png", dpi=150)
plt.close()

# ==========================================
# A5. POLYNOMIAL REGRESSION
# ==========================================
print("\n=== A5. POLYNOMIAL REGRESSION ===")
poly_rows = []
for d in [1, 2, 3, 4]:
    poly = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    poly.fit(X_train, y_train)
    p = poly[0].n_output_features_
    
    tr_pred = poly.predict(X_train)
    te_pred = poly.predict(X_test)
    
    tr_r2 = r2_score(y_train, tr_pred)
    te_r2 = r2_score(y_test, te_pred)
    
    poly_rows.append({
        "Degree": d,
        "Features (p)": p,
        "Train RMSE": np.sqrt(mean_squared_error(y_train, tr_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, te_pred)),
        "Train R2": tr_r2,
        "Train Adj R2": adj_r2(tr_r2, len(X_train), p) if len(X_train) > p + 1 else np.nan,
        "Test R2": te_r2
    })

poly_df = pd.DataFrame(poly_rows)
print(poly_df.round(4).to_string(index=False))

# ==========================================
# A6. CLASSIFICATION: KNN AND DECISION TREE
# ==========================================
print("\n=== A6. CLASSIFICATION ===")
y_class = (df["strength"] >= 40).astype(int)
print("Class proportions:\n", y_class.value_counts(normalize=True).round(4))

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# 1) KNN Sweep
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_train_acc, knn_test_acc = [], []
for k in k_values:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(Xc_train, c_train)
    knn_train_acc.append(knn.score(Xc_train, c_train))
    knn_test_acc.append(knn.score(Xc_test, c_test))

knn_df = pd.DataFrame({
    "K": k_values,
    "Train Acc": knn_train_acc,
    "Test Acc": knn_test_acc
})
print("\nKNN Sweep:")
print(knn_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_values, knn_train_acc, marker="o", label="Train Accuracy")
ax.plot(k_values, knn_test_acc, marker="s", label="Test Accuracy")
ax.set_xscale("log")
ax.set_xticks(k_values)
ax.set_xticklabels(k_values)
ax.set_xlabel("K (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Accuracy vs K")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A6_knn_sweep.png", dpi=150)
plt.close()

best_k = 5
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(Xc_train, c_train)
knn_unscaled = KNeighborsClassifier(n_neighbors=best_k).fit(Xc_train, c_train)
print(f"\nAt K={best_k} -> Scaled Test Acc: {knn_scaled.score(Xc_test, c_test):.4f} | Unscaled Test Acc: {knn_unscaled.score(Xc_test, c_test):.4f}")

# 5 splits stability check
split_accs = []
for rs in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(Xtr, ytr)
    split_accs.append(m.score(Xte, yte))
print(f"5 splits test accuracies (rs=0..4): {[round(a, 4) for a in split_accs]}")
print(f"Mean: {np.mean(split_accs):.4f}, Std (spread): {np.std(split_accs):.4f}, Range: {max(split_accs)-min(split_accs):.4f}")

# 2) Decision Tree Sweep
depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_rows = []
for d in depths:
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0)
    dt.fit(Xc_train, c_train)
    tree_rows.append({
        "max_depth": str(d),
        "Leaves": dt.get_n_leaves(),
        "Train Acc": dt.score(Xc_train, c_train),
        "Test Acc": dt.score(Xc_test, c_test)
    })

tree_df = pd.DataFrame(tree_rows)
print("\nDecision Tree Sweep:")
print(tree_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
x_labels = [str(d) for d in depths]
ax.plot(x_labels, tree_df["Train Acc"], marker="o", label="Train Accuracy")
ax.plot(x_labels, tree_df["Test Acc"], marker="s", label="Test Accuracy")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree Accuracy vs max_depth")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A6_tree_sweep.png", dpi=150)
plt.close()

best_depth = 4
best_tree = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=0).fit(Xc_train, c_train)
print(f"\nTop 2 levels of Decision Tree (max_depth={best_depth}):")
print(export_text(best_tree, feature_names=list(X.columns), max_depth=2))

# ==========================================
# A7. THE METRICS THAT MATTER
# ==========================================
print("\n=== A7. CLASSIFICATION METRICS & THRESHOLDS ===")
def get_clf_metrics(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": name,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

metrics_table = pd.DataFrame([
    get_clf_metrics(f"KNN (K={best_k})", c_test, knn_scaled.predict(Xc_test)),
    get_clf_metrics(f"DecisionTree (depth={best_depth})", c_test, best_tree.predict(Xc_test))
])
print(metrics_table.round(4).to_string(index=False))
metrics_table.to_csv("results/A_metrics.csv", index=False)
print("Saved results/A_metrics.csv")

# Threshold sweep for KNN
proba_knn = knn_scaled.predict_proba(Xc_test)[:, 1]
thresh_rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    preds_t = (proba_knn >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, preds_t).ravel()
    thresh_rows.append({
        "Threshold": t,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision_score(c_test, preds_t, zero_division=0),
        "Recall": recall_score(c_test, preds_t, zero_division=0)
    })
print("\nThreshold Sweep (KNN):")
print(pd.DataFrame(thresh_rows).round(4).to_string(index=False))

# ==========================================
# A8. FEATURE IMPORTANCE
# ==========================================
print("\n=== A8. FEATURE IMPORTANCE ===")
std_pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
perm_knn = permutation_importance(knn_scaled, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
perm_tree = permutation_importance(best_tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Raw Coef": lin_reg.coef_,
    "Std Coef": std_pipe[-1].coef_,
    "Tree Imp": best_tree.feature_importances_,
    "Perm Imp (KNN)": perm_knn.importances_mean,
    "Perm Imp (Tree)": perm_tree.importances_mean
})
print(imp_df.round(4).to_string(index=False))

# Bar chart for 3 importances
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].barh(imp_df["Feature"], np.abs(imp_df["Std Coef"]), color="teal")
axes[0].set_title("|Standardised Linear Coef|")

axes[1].barh(imp_df["Feature"], imp_df["Tree Imp"], color="coral")
axes[1].set_title("Decision Tree Importance")

axes[2].barh(imp_df["Feature"], imp_df["Perm Imp (KNN)"], color="slateblue")
axes[2].set_title("Permutation Importance (KNN F1)")

plt.tight_layout()
plt.savefig("figures/A8_importance.png", dpi=150)
plt.close()

# ==========================================
# BONUS: PHYSICS FEATURES
# ==========================================
print("\n=== BONUS: PHYSICS FEATURES ===")
X_phys1 = X.copy()
X_phys1["age"] = np.log(X_phys1["age"])
Xp1_tr, Xp1_te, yp_tr, yp_te = train_test_split(X_phys1, y, test_size=0.2, random_state=42)
m_phys1 = LinearRegression().fit(Xp1_tr, yp_tr)
print(f"1) With log(age) (p={Xp1_tr.shape[1]}): Test R2 = {r2_score(yp_te, m_phys1.predict(Xp1_te)):.4f}")

X_phys2 = X_phys1.copy()
X_phys2["w_b_ratio"] = X_phys2["water"] / (X_phys2["cement"] + X_phys2["slag"] + X_phys2["ash"])
X_phys2 = X_phys2.drop(columns=["water"])
Xp2_tr, Xp2_te, _, _ = train_test_split(X_phys2, y, test_size=0.2, random_state=42)
m_phys2 = LinearRegression().fit(Xp2_tr, yp_tr)
print(f"2) With log(age) & w/b ratio (p={Xp2_tr.shape[1]}): Test R2 = {r2_score(yp_te, m_phys2.predict(Xp2_te)):.4f}")