"""
train_model.py
Zero Trust Network Access - Insider Threat Detection
Trains a Random Forest classifier on behavioral data and saves model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("=" * 55)
print("  Zero Trust - Insider Threat Model Training")
print("=" * 55)

# ── 1. Load dataset ──────────────────────────────────────
df = pd.read_csv("dataset.csv")
print(f"\n[INFO] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"       Normal (0): {(df['label']==0).sum()} | Threat (1): {(df['label']==1).sum()}")

# ── 2. Features & labels ─────────────────────────────────
FEATURES = [
    "login_hour",
    "failed_logins",
    "data_transferred_mb",
    "unusual_location",
    "after_hours_access",
    "privilege_escalation",
    "multiple_sessions",
    "request_rate",
]

X = df[FEATURES]
y = df["label"]

# ── 3. Train / test split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Scale features ────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train model ───────────────────────────────────────
print("\n[INFO] Training Random Forest classifier …")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_scaled, y_train)

# ── 6. Evaluate ──────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\n[RESULT] Accuracy : {acc * 100:.2f}%")
print("\n[RESULT] Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Threat"]))

print("[RESULT] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"         TN={cm[0,0]}  FP={cm[0,1]}")
print(f"         FN={cm[1,0]}  TP={cm[1,1]}")

# ── 7. Feature importance ────────────────────────────────
print("\n[INFO] Feature Importance:")
importance = sorted(
    zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True
)
for feat, score in importance:
    bar = "█" * int(score * 40)
    print(f"  {feat:<25} {bar} {score:.3f}")

# ── 8. Save model + scaler ───────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": FEATURES}, f)

print("\n[✓] model.pkl saved successfully.")
print("=" * 55)