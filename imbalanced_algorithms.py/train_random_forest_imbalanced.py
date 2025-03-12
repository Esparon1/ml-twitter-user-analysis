import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

#  Charger les nouveaux datasets déséquilibrés
X_train = pd.read_csv("../X_train_imbalanced.csv")
X_test = pd.read_csv("../X_test_imbalanced.csv")
y_train = pd.read_csv("../y_train_imbalanced.csv").values.ravel()
y_test = pd.read_csv("../y_test_imbalanced.csv").values.ravel()

print("🚀 Training Random Forest model on imbalanced data...")

#  Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Évaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Legitimate", "Polluter"])
roc_auc = roc_auc_score(y_test, y_proba)

tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

#  Affichage des résultats
print(f" Random Forest (Imbalanced) Accuracy: {accuracy:.4f}")
print(f" True Positive Rate (TPR): {tpr:.4f}")
print(f" False Positive Rate (FPR): {fpr:.4f}")
print(f" F1-score for Polluters: {class_report.split()[-2]}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

#  Sauvegarde des résultats
results = pd.DataFrame({
    "Model": ["Random Forest (Imbalanced)"],
    "Accuracy": [accuracy],
    "TPR": [tpr],
    "FPR": [fpr],
    "F1-score": [float(class_report.split()[-2])],
    "ROC-AUC": [roc_auc]
})
results.to_csv("../model_results_imbalanced.csv", index=False, mode='a', header=False)
