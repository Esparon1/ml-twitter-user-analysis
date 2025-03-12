import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model results
models = ["Decision Tree", "Bagging", "GBoost", "AdaBoost", "Random Forest", "Naïve Bayes"]
accuracy = [0.9328, 0.9573, 0.9572, 0.9341, 0.9592, 0.5960]
tpr = [0.9319, 0.9643, 0.9638, 0.9288, 0.9663, 0.2770]
fpr = [0.0662, 0.0508, 0.0505, 0.0597, 0.0492, 0.0299]
f1_score = [0.93, 0.96, 0.96, 0.93, 0.96, 0.55]
roc_auc = [0.9328, 0.9898, 0.9909, 0.9809, 0.9918, 0.8793]

# Bar chart - Accuracy Comparison
plt.figure(figsize=(10, 5))
plt.bar(models, accuracy, color=['blue', 'green', 'red', 'purple', 'orange', 'gray'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Bar chart - F1-Score, TPR, and FPR
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, tpr, width, label="True Positive Rate (TPR)", color="green")
ax.bar(x, fpr, width, label="False Positive Rate (FPR)", color="red")
ax.bar(x + width, f1_score, width, label="F1-Score", color="blue")

ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
