import pandas as pd
from sklearn.model_selection import train_test_split




data = pd.read_csv("final_dataset_updated.csv")


X = data.drop(columns=["Label"])  # Features
y = data["Label"]  # Target variable

print("🚀 Splitting dataset...")

# Split (80%) (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the train and test datasets as CSV files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Train/Test datasets saved successfully!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
