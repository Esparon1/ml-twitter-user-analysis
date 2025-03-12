import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset_5pct_pollueurs.csv")

X = data.drop(columns=["Label"])  
y = data["Label"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Save the train and test datasets as CSV files
X_train.to_csv("X_train_imbalanced.csv", index=False)
X_test.to_csv("X_test_imbalanced.csv", index=False)
y_train.to_csv("y_train_imbalanced.csv", index=False)
y_test.to_csv("y_test_imbalanced.csv", index=False)


print(data.head())
