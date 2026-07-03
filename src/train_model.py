import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

# Convert labels into Normal(0) and Attack(1)
data[41] = data[41].apply(lambda x: 0 if x == "normal" else 1)

# Encode categorical columns
encoder = LabelEncoder()

for col in [1, 2, 3]:
    data[col] = encoder.fit_transform(data[col])

# Features and target
X = data.drop(columns=[41, 42])
y = data[41]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, "models/ids_model.pkl")

print("\nModel saved successfully!")