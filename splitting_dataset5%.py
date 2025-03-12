import pandas as pd

data = pd.read_csv("final_dataset_updated.csv")

polluters = data[data["Label"] == 1]
legitimate_users = data[data["Label"] == 0]

polluters_subset = polluters.sample(frac=0.05, random_state=42)

imbalanced_data = pd.concat([legitimate_users, polluters_subset], ignore_index=True)

imbalanced_data.to_csv("dataset_5pct_pollueurs.csv", index=False)

print("saved as 'dataset_5pct_pollueurs.csv'")
print(imbalanced_data["Label"].value_counts(normalize=True) * 100)
