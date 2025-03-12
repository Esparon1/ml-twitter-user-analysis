import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the updated dataset
df = pd.read_csv("final_dataset_updated.csv")

print(f"Dataset shape: {df.shape}")

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nChecking for new features:")
print(df[["Hashtag_Ratio", "Time_Between_Variability"]].head())

print("\nClass distribution (Label column):")
print(df["Label"].value_counts(normalize=True))


df.iloc[:, :-1].hist(figsize=(12, 8), bins=30)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()