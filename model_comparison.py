import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])
}

results = []

fig, ax = plt.subplots(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc
    })

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax,
        name=f"{name} (AUC = {auc:.3f})"
    )

ax.set_title("ROC Curve Comparison of Different Models")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_comparison_fixed.png", dpi=300)
plt.close()

df_results = pd.DataFrame(results)
df_results.to_csv("model_comparison_results.csv", index=False)

print(df_results)