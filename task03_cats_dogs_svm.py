# ============================================================
# TASK 03 – Cats vs Dogs Image Classification using SVM
# Dataset: Kaggle Dogs vs Cats dataset
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# HOW TO USE WITH THE REAL KAGGLE DATASET
# ─────────────────────────────────────────────────────────────
# 1. Download from: https://www.kaggle.com/c/dogs-vs-cats/data
# 2. Extract so you have:  train/cats/*.jpg  and  train/dogs/*.jpg
# 3. Set USE_REAL_DATA = True and update DATASET_PATH below
# ─────────────────────────────────────────────────────────────

USE_REAL_DATA = False           # ← Set True when you have the files
DATASET_PATH  = "train"         # folder containing cats/ and dogs/
IMG_SIZE      = (64, 64)        # resize all images to this
MAX_PER_CLASS = 1000            # limit to speed up SVM training

# ─────────────────────────────────────────────
# Helper: load images from disk
# ─────────────────────────────────────────────
def load_images(dataset_path, img_size, max_per_class):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow not installed. Run: pip install Pillow")

    X, y = [], []
    for label, class_name in enumerate(["cats", "dogs"]):
        folder = os.path.join(dataset_path, class_name)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files  = files[:max_per_class]
        print(f"  Loading {len(files)} '{class_name}' images …")
        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                img = img.resize(img_size)
                X.append(np.array(img).flatten())
                y.append(label)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 03 – Cats vs Dogs Classification (SVM)")
print("=" * 55)

if USE_REAL_DATA:
    print("\nLoading real Kaggle images …")
    X, y = load_images(DATASET_PATH, IMG_SIZE, MAX_PER_CLASS)
else:
    # ── Synthetic stand-in ──────────────────────────────────
    # Each 'image' is IMG_SIZE[0]*IMG_SIZE[1]*3 random pixels.
    # Class 0 = cats (slightly darker mean), Class 1 = dogs.
    print("\n[Demo mode] Generating synthetic image data …")
    print("(Set USE_REAL_DATA=True to use the Kaggle dataset)")
    np.random.seed(42)
    n_each = 500
    cats = np.random.normal(loc=100, scale=40,
                            size=(n_each, IMG_SIZE[0]*IMG_SIZE[1]*3)).clip(0, 255)
    dogs = np.random.normal(loc=140, scale=40,
                            size=(n_each, IMG_SIZE[0]*IMG_SIZE[1]*3)).clip(0, 255)
    X = np.vstack([cats, dogs]).astype(np.float32)
    y = np.array([0]*n_each + [1]*n_each)

print(f"\nDataset shape : {X.shape}")
print(f"Labels        : {np.bincount(y)}  (0=cat, 1=dog)")

# ─────────────────────────────────────────────
# 2. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. Normalise pixel values  [0, 255] → [0, 1]
# ─────────────────────────────────────────────
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# ─────────────────────────────────────────────
# 4. PCA – reduce dimensionality (speeds up SVM)
# ─────────────────────────────────────────────
N_COMPONENTS = 100   # keep top 100 principal components
print(f"\nApplying PCA (n_components={N_COMPONENTS}) …")
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print(f"Variance explained : {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────────────────────────────────────
# 5. Scale PCA features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_pca)
X_test_sc  = scaler.transform(X_test_pca)

# ─────────────────────────────────────────────
# 6. Train SVM  (RBF kernel)
# ─────────────────────────────────────────────
print("\nTraining SVM (RBF kernel) …")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, probability=True)
svm.fit(X_train_sc, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────
y_pred = svm.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)

print("\n--- Evaluation Results ---")
print(f"  Accuracy : {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# ─────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 03 – Cats vs Dogs SVM Classifier", fontsize=14, fontweight="bold")

# (a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion Matrix  (Accuracy={acc*100:.1f}%)")

# (b) PCA explained variance
axes[1].plot(np.cumsum(pca.explained_variance_ratio_) * 100, color="steelblue")
axes[1].axhline(95, color="red", linestyle="--", label="95% variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("PCA – Explained Variance")
axes[1].legend()

plt.tight_layout()
plt.savefig("task03_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → task03_results.png")

# ─────────────────────────────────────────────
# 9. Optional: GridSearchCV for hyperparameter tuning
# ─────────────────────────────────────────────
print("\n--- Optional: GridSearchCV (uncomment to run) ---")
# param_grid = {"C": [1, 10, 100], "gamma": ["scale", "auto"]}
# grid = GridSearchCV(SVC(kernel="rbf", random_state=42),
#                     param_grid, cv=5, n_jobs=-1, verbose=1)
# grid.fit(X_train_sc, y_train)
# print(f"Best params : {grid.best_params_}")
# print(f"Best CV acc : {grid.best_score_*100:.2f}%")
