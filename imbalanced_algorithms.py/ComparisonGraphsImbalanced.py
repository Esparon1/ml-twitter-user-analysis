import matplotlib.pyplot as plt
import numpy as np

models_imbalanced = ["Decision Tree", "Bagging", "GBoost", "AdaBoost", "Random Forest", "Naïve Bayes"]
accuracy_imbalanced = [0.9789, 0.9789, 0.9794, 0.9558, 0.9789, 0.9321]
tpr_imbalanced = [0.7252, 0.6847, 0.7162, 0.2252, 0.6847, 0.2703]
fpr_imbalanced = [0.0065, 0.0042, 0.0054, 0.0021, 0.0042, 0.0298]
f1_score_imbalanced = [0.98, 0.98, 0.98, 0.94, 0.98, 0.93]
roc_auc_imbalanced = [0.9745, 0.9842, 0.9875, 0.9678, 0.9842, 0.8695]

#  Bar chart - Comparaison des Accuracies
plt.figure(figsize=(10, 5))
plt.bar(models_imbalanced, accuracy_imbalanced, color=['blue', 'green', 'red', 'purple', 'orange', 'gray'])
plt.xlabel("Modèles")
plt.ylabel("Accuracy")
plt.title("Comparaison des Accuracies des modèles (Données Déséquilibrées)")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ✅ Bar chart - Comparaison des TPR, FPR et F1-Score
x = np.arange(len(models_imbalanced))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, tpr_imbalanced, width, label="True Positive Rate (TPR)", color="green")
ax.bar(x, fpr_imbalanced, width, label="False Positive Rate (FPR)", color="red")
ax.bar(x + width, f1_score_imbalanced, width, label="F1-Score", color="blue")

ax.set_xlabel("Modèles")
ax.set_ylabel("Scores")
ax.set_title("Comparaison des performances des modèles (Données Déséquilibrées)")
ax.set_xticks(x)
ax.set_xticklabels(models_imbalanced, rotation=30)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
