# ============================================================
# TASK 02 – Customer Segmentation using K-Means Clustering
# Dataset: Mall Customer Dataset
# Features: Annual Income, Spending Score, Age
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────


np.random.seed(42)
n = 200

gender         = np.random.choice(["Male", "Female"], n)
age            = np.random.randint(18, 70, n)
annual_income  = np.random.randint(15, 140, n)   # in k$

# Create natural clusters in spending score
spending_score = np.where(
    annual_income < 40,  np.random.randint(1,  40, n),
    np.where(
        annual_income > 100, np.random.randint(60, 100, n),
        np.random.randint(20, 80, n)
    )
)

df = pd.DataFrame({
    "gender"        : gender,
    "age"           : age,
    "annual_income" : annual_income,
    "spending_score": spending_score
})

print("=" * 55)
print("  TASK 02 – Customer Segmentation (K-Means)")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2. Feature Selection & Scaling
# ─────────────────────────────────────────────
features = ["annual_income", "spending_score"]   # core 2-D view
X = df[features].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 3. Elbow Method – find optimal K
# ─────────────────────────────────────────────
inertia    = []
sil_scores = []
K_range    = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_sc)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sc, km.labels_))

best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K (highest silhouette score) : {best_k}")

# ─────────────────────────────────────────────
# 4. Final K-Means Model
# ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X_sc)
df["cluster"] = kmeans.labels_

sil  = silhouette_score(X_sc, kmeans.labels_)
db   = davies_bouldin_score(X_sc, kmeans.labels_)

print("\n--- Evaluation Metrics ---")
print(f"  Silhouette Score      : {sil:.4f}  (higher = better, max 1)")
print(f"  Davies-Bouldin Score  : {db:.4f}   (lower  = better)")
print("\n--- Cluster Sizes ---")
print(df["cluster"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────
# 5. Cluster Profile
# ─────────────────────────────────────────────
print("\n--- Cluster Profiles (mean values) ---")
profile = df.groupby("cluster")[["age", "annual_income", "spending_score"]].mean().round(2)
print(profile)

# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
colors = cm.tab10(np.linspace(0, 1, best_k))

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Task 02 – Customer Segmentation (K-Means Clustering)", fontsize=14, fontweight="bold")

# (a) Elbow Curve
axes[0].plot(list(K_range), inertia, "bo-", markersize=7)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
axes[0].legend()

# (b) Silhouette Scores
axes[1].plot(list(K_range), sil_scores, "gs-", markersize=7)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Scores")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
axes[1].legend()

# (c) Cluster Scatter Plot
for c in range(best_k):
    mask = df["cluster"] == c
    axes[2].scatter(
        df.loc[mask, "annual_income"],
        df.loc[mask, "spending_score"],
        label=f"Cluster {c}",
        alpha=0.7, s=50,
        color=colors[c]
    )

# Mark centroids (inverse-transform)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
axes[2].scatter(
    centroids_orig[:, 0], centroids_orig[:, 1],
    c="black", s=180, marker="X", zorder=5, label="Centroids"
)
axes[2].set_xlabel("Annual Income (k$)")
axes[2].set_ylabel("Spending Score (1-100)")
axes[2].set_title(f"Customer Clusters (K={best_k})")
axes[2].legend()

plt.tight_layout()
plt.savefig("task02_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → task02_results.png")
