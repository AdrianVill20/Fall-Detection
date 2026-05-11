"""
train.py
Trains a Random Forest classifier on extracted keypoints.
Includes augmentation on fall samples to improve accuracy.
Usage: python model/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH   = "data/keypoints.csv"
MODEL_PATH = "model/fall_model.joblib"
NOISE_STD  = 0.01   # small noise for augmentation
AUG_TIMES  = 3      # how many extra copies of fall samples to generate


def augment_falls(df):
    """Add slightly noisy copies of fall samples to boost the fall class."""
    fall_df = df[df.label == 1]
    feature_cols = [c for c in df.columns if c != "label"]
    augmented = []

    for _ in range(AUG_TIMES):
        noisy = fall_df.copy()
        noisy[feature_cols] = noisy[feature_cols] + np.random.normal(
            0, NOISE_STD, noisy[feature_cols].shape
        )
        augmented.append(noisy)

    combined = pd.concat([df] + augmented, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return combined


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)

    print(f"Before augmentation — Fall: {len(df[df.label==1])} | No-fall: {len(df[df.label==0])}")
    df = augment_falls(df)
    print(f"After augmentation  — Fall: {len(df[df.label==1])} | No-fall: {len(df[df.label==0])}")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",   # extra safety for imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["No Fall", "Fall"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
