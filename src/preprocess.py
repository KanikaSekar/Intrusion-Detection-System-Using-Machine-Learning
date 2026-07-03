import pandas as pd
data = pd.read_csv('dataset/KDDTrain+.txt', header=None)
print("Attack categories:/n")
print(data[41].value_counts())
print("/nTotal categories:", data[41].nunique())