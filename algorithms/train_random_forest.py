import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

print(" Loading datasets...")

#  Load training and testing datasets
try:
    X_train = pd.read_csv("../X_train.csv")
    X_test = pd.read_csv("../X_test.csv")
    y_train = pd.read_csv("../y_train.csv").values.ravel()  # Convert to 1D array
    y_test = pd.read_csv("../y_test.csv").values.ravel()
    print(f" Datasets loaded successfully! Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")
except Exception as e:
    print(f" Error loading datasets: {e}")
    exit()

print(" Training Random Forest model...")
try:
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
    model.fit(X_train, y_train)
    print(" Model training completed!")
except Exception as e:
    print(f" Error training the model: {e}")
    exit()

print("Making predictions...")
try:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC
    print(" Predictions completed!")
except Exception as e:
    print(f" Error in predictions: {e}")
    exit()

print(" Evaluating model...")
try:
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Legitimate", "Polluter"])
    roc_auc = roc_auc_score(y_test, y_proba)

    # Extract TPR and FPR
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate

    # ✅ Print results
    print(f" Random Forest Accuracy: {accuracy:.4f}")
    print(f" True Positive Rate (TPR): {tpr:.4f}")
    print(f" False Positive Rate (FPR): {fpr:.4f}")
    print(f" F1-score for Polluters: {class_report.split()[-2]}")
    print(f" ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f" Error in evaluation: {e}")
    exit()

print("Saving results...")
try:
    results = pd.DataFrame({
        "Model": ["Random Forest"],
        "Accuracy": [accuracy],
        "TPR": [tpr],
        "FPR": [fpr],
        "F1-score": [float(class_report.split()[-2])],
        "ROC-AUC": [roc_auc]
    })
    results.to_csv("../model_results.csv", index=False, mode='a', header=False)
    print("✅ Results saved successfully!")
except Exception as e:
    print(f" Error saving results: {e}")
    exit()

print(" Script completed successfully!")
