import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load training data
train_data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

train_data[41] = train_data[41].apply(
    lambda x: 0 if x == "normal" else 1
)

train_data = pd.get_dummies(
    train_data,
    columns=[1, 2, 3]
)

train_data.columns = train_data.columns.astype(str)

X_train = train_data.drop(columns=["41", "42"])

# Load trained model
model = joblib.load("models/gradient_boosting.pkl")

# Get feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Top 10 features
top10 = feature_importance.head(10)

plt.figure(figsize=(10,6))
plt.barh(top10["Feature"], top10["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("screenshots/feature_importance.png")

print("Feature Importance graph saved successfully!")
