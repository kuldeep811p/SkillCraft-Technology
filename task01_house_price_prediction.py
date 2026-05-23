# ============================================================
# TASK 01 - House Price Prediction using Linear Regression
# Dataset: House Price Dataset
# Features: Square Footage, Bedrooms, Bathrooms
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────


np.random.seed(42)
n = 500

sqft       = np.random.randint(500, 5000, n)
bedrooms   = np.random.randint(1, 6, n)
bathrooms  = np.random.randint(1, 4, n)

# Price roughly correlated with features + noise
price = (
    100 * sqft
    + 15000 * bedrooms
    + 10000 * bathrooms
    + np.random.normal(0, 20000, n)
    + 50000          # base price
)

df = pd.DataFrame({
    "sqft"      : sqft,
    "bedrooms"  : bedrooms,
    "bathrooms" : bathrooms,
    "price"     : price
})

print("=" * 55)
print("  TASK 01 – House Price Linear Regression")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2. Feature & Target Split
# ─────────────────────────────────────────────
X = df[["sqft", "bedrooms", "bathrooms"]]
y = df["price"]

# ─────────────────────────────────────────────
# 3. Train / Test Split  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. Feature Scaling  (optional but good practice)
# ─────────────────────────────────────────────
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. Train Linear Regression Model
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("\n--- Model Coefficients ---")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:12s}: {coef:>12.2f}")
print(f"  {'Intercept':12s}: {model.intercept_:>12.2f}")

# ─────────────────────────────────────────────
# 6. Predictions & Evaluation
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R²   : {r2:.4f}")

# ─────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 01 – House Price Prediction (Linear Regression)", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.4)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# (b) Residuals Distribution
residuals = y_test - y_pred
axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residuals Distribution")

# (c) Feature Correlation Heatmap
axes[2].set_title("Feature Correlation Heatmap")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[2], square=True, linewidths=0.5)

plt.tight_layout()
plt.savefig("task01_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → task01_results.png")

# ─────────────────────────────────────────────
# 8. Sample Prediction
# ─────────────────────────────────────────────
sample = pd.DataFrame({"sqft": [2000], "bedrooms": [3], "bathrooms": [2]})
sample_sc = scaler.transform(sample)
pred_price = model.predict(sample_sc)[0]
print(f"\nSample prediction → sqft=2000, bedrooms=3, bathrooms=2")
print(f"  Predicted price : ${pred_price:,.2f}")
