# Machine Learning Internship – Task Submissions

## Overview

| Task | Title | Algorithm | Dataset |
|------|-------|-----------|---------|
| 01 | House Price Prediction | Linear Regression | House Price Dataset |
| 02 | Customer Segmentation | K-Means Clustering | Mall Customer Dataset |
| 03 | Cats vs Dogs Classification | Support Vector Machine (SVM) | Kaggle Dogs vs Cats |
| 04 | Hand Gesture Recognition | CNN (Deep Learning) | Hand Gesture Dataset |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
```

---

## Task 01 – House Price Prediction (Linear Regression)

**File:** `task01_house_price_prediction.py`

**What it does:**
- Predicts house prices from square footage, bedrooms, and bathrooms
- Applies StandardScaler for feature normalisation
- Evaluates with MAE, RMSE, and R² score
- Plots: Actual vs Predicted, Residuals Distribution, Feature Correlation Heatmap

**To use real data:** Replace the synthetic data block with:
```python
df = pd.read_csv("house_prices.csv")
```

---

## Task 02 – Customer Segmentation (K-Means Clustering)

**File:** `task02_customer_segmentation.py`

**What it does:**
- Groups retail customers by Annual Income and Spending Score
- Uses the Elbow Method + Silhouette Score to find the optimal K
- Evaluates with Silhouette Score and Davies-Bouldin Score
- Plots: Elbow Curve, Silhouette Scores, Cluster Scatter Plot

**To use real data:**
```python
df = pd.read_csv("Mall_Customers.csv")
```

---

## Task 03 – Cats vs Dogs (SVM Classifier)

**File:** `task03_cats_dogs_svm.py`

**What it does:**
- Classifies images as cat or dog using an RBF kernel SVM
- Applies PCA (100 components) to reduce 64×64×3 image features
- Evaluates with accuracy, precision, recall, F1
- Plots: Confusion Matrix, PCA Variance Curve

**To use real data:**
1. Download from: https://www.kaggle.com/c/dogs-vs-cats/data
2. Set `USE_REAL_DATA = True` in the script
3. Update `DATASET_PATH` to point at your `train/` folder

---

## Task 04 – Hand Gesture Recognition (CNN)

**File:** `task04_hand_gesture_recognition.py`

**What it does:**
- Multi-class classification of hand gestures from images
- 3-block CNN with BatchNorm, Dropout, EarlyStopping
- Evaluates with accuracy, classification report, confusion matrix
- Plots: Accuracy curve, Loss curve, Confusion Matrix
- Saves model to `task04_gesture_model.h5`

**To use real data:**
1. Download from: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
2. Set `USE_REAL_DATA = True` in the script
3. Update `DATASET_PATH` to your dataset folder

---

## Running the Scripts

```bash
python task01_house_price_prediction.py
python task02_customer_segmentation.py
python task03_cats_dogs_svm.py
python task04_hand_gesture_recognition.py
```

Each script saves its output plot as a PNG in the same directory.
