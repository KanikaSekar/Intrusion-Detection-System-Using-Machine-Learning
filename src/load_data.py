import pandas as pd
data = pd.read_csv("dataset/KDDTrain+.txt", header=None)
print("Data Loaded Successfully!")
print("Data shape:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())