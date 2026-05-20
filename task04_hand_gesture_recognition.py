# ============================================================
# TASK 04 – Hand Gesture Recognition
# Dataset: Hand Gesture Recognition Dataset (Kaggle)
# Model  : CNN (TensorFlow / Keras)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────
# HOW TO USE WITH THE REAL DATASET
# ─────────────────────────────────────────────────────────────
# 1. Download from Kaggle:
#    https://www.kaggle.com/datasets/gti-upm/leapgestrecog
#    or https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset
# 2. Set USE_REAL_DATA = True and update DATASET_PATH
#    Expected structure: DATASET_PATH/<gesture_name>/*.png
# ─────────────────────────────────────────────────────────────

USE_REAL_DATA = False          # ← Set True with the real dataset
DATASET_PATH  = "leapGestRecog"
IMG_SIZE      = (64, 64)
MAX_PER_CLASS = 300
EPOCHS        = 20
BATCH_SIZE    = 32

# ─────────────────────────────────────────────
# 1.  TensorFlow / Keras import
# ─────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print(f"TensorFlow version : {tf.__version__}")
except ImportError:
    raise ImportError(
        "TensorFlow not installed.\n"
        "Run: pip install tensorflow\n"
        "For GPU: pip install tensorflow[and-cuda]"
    )

# ─────────────────────────────────────────────
# Helper: load images from sub-folders
# ─────────────────────────────────────────────
def load_gesture_images(root, img_size, max_per_class):
    from PIL import Image
    X, y, class_names = [], [], []
    gesture_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    for label, gname in enumerate(gesture_dirs):
        class_names.append(gname)
        folder = os.path.join(root, gname)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_per_class]
        print(f"  '{gname}': {len(files)} images")
        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                img = img.resize(img_size)
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(y), class_names


# ─────────────────────────────────────────────
# 2. Load / Generate Data
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 04 – Hand Gesture Recognition (CNN)")
print("=" * 55)

NUM_CLASSES = 10   # change if your dataset has different count

if USE_REAL_DATA:
    print(f"\nLoading images from '{DATASET_PATH}' …")
    X, y, class_names = load_gesture_images(DATASET_PATH, IMG_SIZE, MAX_PER_CLASS)
    NUM_CLASSES = len(class_names)
else:
    print("\n[Demo mode] Generating synthetic gesture data …")
    print("(Set USE_REAL_DATA=True and point DATASET_PATH at your dataset)")
    np.random.seed(42)
    n_each = 200
    class_names = [f"gesture_{i}" for i in range(NUM_CLASSES)]
    X_list, y_list = [], []
    for c in range(NUM_CLASSES):
        imgs = np.random.normal(
            loc=c * 20, scale=30,
            size=(n_each, IMG_SIZE[0], IMG_SIZE[1], 3)
        ).clip(0, 255)
        X_list.append(imgs)
        y_list.extend([c] * n_each)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)

print(f"\nTotal samples  : {len(X)}")
print(f"Image shape    : {X.shape[1:]}")
print(f"Classes ({NUM_CLASSES})  : {class_names}")

# ─────────────────────────────────────────────
# 3. Normalise & One-Hot Encode
# ─────────────────────────────────────────────
X = X / 255.0
y_cat = to_categorical(y, NUM_CLASSES)

# ─────────────────────────────────────────────
# 4. Train / Validation / Test Split
# ─────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y_cat, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42,
    stratify=np.argmax(y_tmp, axis=1)
)

print(f"\nTrain  : {len(X_train)}")
print(f"Val    : {len(X_val)}")
print(f"Test   : {len(X_test)}")

# ─────────────────────────────────────────────
# 5. Build CNN Model
# ─────────────────────────────────────────────
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Fully connected head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_cnn(X_train.shape[1:], NUM_CLASSES)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ─────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

print(f"\nTraining for up to {EPOCHS} epochs …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- Test Results ---")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save model
model.save("task04_gesture_model.h5")
print("\nModel saved → task04_gesture_model.h5")

# ─────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Task 04 – Hand Gesture Recognition (CNN)", fontsize=14, fontweight="bold")

# (a) Training curves – Accuracy
axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="steelblue")
axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="coral")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy over Epochs"); axes[0].legend()

# (b) Training curves – Loss
axes[1].plot(history.history["loss"],     label="Train Loss", color="steelblue")
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="coral")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title("Loss over Epochs"); axes[1].legend()

# (c) Confusion Matrix (first 10 classes shown)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=axes[2], cmap="Blues", colorbar=False,
          xticks_rotation=45)
axes[2].set_title(f"Confusion Matrix  (Acc={test_acc*100:.1f}%)")

plt.tight_layout()
plt.savefig("task04_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → task04_results.png")

# ─────────────────────────────────────────────
# 9. Inference on a single image
# ─────────────────────────────────────────────
def predict_gesture(image_array):
    """
    image_array : H×W×3 uint8 numpy array (raw image)
    Returns     : predicted class name
    """
    img = image_array.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # add batch dim
    probs = model.predict(img, verbose=0)[0]
    return class_names[np.argmax(probs)], float(np.max(probs))

# Demo call
sample_img = (X_test[0] * 255).astype(np.uint8)
pred_class, confidence = predict_gesture(sample_img)
print(f"\nSample inference → Class: '{pred_class}'  Confidence: {confidence*100:.1f}%")
