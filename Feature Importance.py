import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("top10_features.csv", header=None)
df.columns = ["Feature", "Coefficient"]

df["Feature"] = df["Feature"].astype(str)
df["Coefficient"] = df["Coefficient"].astype(float)
df["AbsCoeff"] = df["Coefficient"].abs()

# Sort by importance
df = df.sort_values("AbsCoeff", ascending=True)

y_pos = range(len(df))

plt.figure(figsize=(6, 4))
plt.barh(y_pos, df["AbsCoeff"])
plt.yticks(y_pos, df["Feature"])

plt.xlabel("Absolute Coefficient Value")
plt.title("Top 10 Important Features (Logistic Regression)")

plt.tight_layout()
plt.savefig("feature_importance_bar.png", dpi=300)
plt.close()