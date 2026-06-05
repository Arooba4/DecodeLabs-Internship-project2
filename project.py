# ============================================================
#   Project 2: Basic Classification Model — Iris Dataset
#   Algorithm: K-Nearest Neighbors (KNN)
# ============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

print("=" * 60)
print("   CLASSIFICATION MODEL — IRIS DATASET (KNN)")
print("=" * 60)

# ── 2. Load & Understand the Dataset ────────────────────────
iris = load_iris()

# Convert to DataFrame for easy exploration
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\n📊 DATASET OVERVIEW")
print("-" * 40)
print(f"Total samples  : {len(df)}")
print(f"Features       : {list(iris.feature_names)}")
print(f"Target classes : {list(iris.target_names)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())

print(f"\nClass distribution:")
print(df['species'].value_counts().to_string())

print(f"\nBasic Statistics:")
print(df.describe().to_string())

# ── 3. Prepare Features & Labels ────────────────────────────
X = iris.data          # Features (4 columns)
y = iris.target        # Labels  (0 = setosa, 1 = versicolor, 2 = virginica)

# ── 4. Split Data into Training & Testing Sets ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # reproducible split
    stratify=y          # keep class balance in both sets
)

print("\n✂️  TRAIN / TEST SPLIT")
print("-" * 40)
print(f"Training samples : {len(X_train)}  ({len(X_train)/len(X)*100:.0f}%)")
print(f"Testing  samples : {len(X_test)}   ({len(X_test)/len(X)*100:.0f}%)")

# ── 5. Feature Scaling ───────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)         # transform test (no fit!)

# ── 6. Train the Model ───────────────────────────────────────
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

print(f"\n🤖 MODEL TRAINED")
print("-" * 40)
print(f"Algorithm  : K-Nearest Neighbors")
print(f"K value    : {k}")

# ── 7. Make Predictions ──────────────────────────────────────
y_pred = model.predict(X_test_scaled)

# ── 8. Evaluate the Model ────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n📈 MODEL PERFORMANCE")
print("-" * 40)
print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm_df = pd.DataFrame(
    cm,
    index   = [f"Actual {n}"    for n in iris.target_names],
    columns = [f"Predicted {n}" for n in iris.target_names]
)
print(cm_df.to_string())

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── 9. Predict a New Sample ──────────────────────────────────
sample = np.array([[5.1, 3.5, 1.4, 0.2]])          # typical setosa
sample_scaled = scaler.transform(sample)
prediction    = model.predict(sample_scaled)
probability   = model.predict_proba(sample_scaled)

print("🌸 PREDICT A NEW FLOWER")
print("-" * 40)
print(f"Input features : sepal length=5.1, sepal width=3.5, "
      f"petal length=1.4, petal width=0.2")
print(f"Predicted class: {iris.target_names[prediction[0]]}")
print("Class probabilities:")
for cls, prob in zip(iris.target_names, probability[0]):
    print(f"  {cls:<15}: {prob*100:.1f}%")

print("\n" + "=" * 60)
print("  Project Complete ✓")
print("=" * 60)