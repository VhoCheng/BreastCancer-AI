import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

def main():
    # 1) Load dataset (built-in, no manual download needed)
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0=malignant, 1=benign in sklearn dataset

    print("Dataset shape:", X.shape)
    print("Classes:", data.target_names)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Build pipeline: scaling + model
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    # 4) Train
    clf.fit(X_train, y_train)

    # 5) Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]  # probability of class 1 (benign)

    # 6) Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 7) Save figures (paper-ready)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot()
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve (Test Set)")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=200)
    plt.close()

    # 8) Simple “engineering-style” output: model coefficients (top features)
    model = clf.named_steps["model"]
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    coefs.head(10).to_csv("top10_features.csv")
    print("\nSaved: confusion_matrix.png, roc_curve.png, top10_features.csv")

if __name__ == "__main__":
    main()