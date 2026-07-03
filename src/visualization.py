import pandas as pd
import matplotlib.pyplot as plt

# Load training data
data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

# Attack vs Normal
attack_counts = data[41].value_counts()

plt.figure(figsize=(10, 6))
attack_counts.head(10).plot(kind="bar")
plt.title("Top 10 Traffic Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("screenshots/attack_distribution.png")

print("Graph saved successfully!")