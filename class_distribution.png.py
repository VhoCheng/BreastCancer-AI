from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
labels = data.target

plt.figure(figsize=(4,4))
plt.hist(labels, bins=2)
plt.xticks([0,1], ["Malignant", "Benign"])
plt.title("Class Distribution of the Dataset")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.close()