import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training data
train_data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

# Convert labels to binary
train_data[41] = train_data[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# Encode categorical columns
encoder = LabelEncoder()

for col in [1, 2, 3]:
    train_data[col] = encoder.fit_transform(train_data[col])

# Features and labels
X_train = train_data.drop(columns=[41, 42])
y_train = train_data[41]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Load test data
test_data = pd.read_csv("dataset/KDDTest+.txt", header=None)

# Convert labels
test_data[41] = test_data[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# Encode categorical columns
for col in [1, 2, 3]:
    test_data[col] = encoder.fit_transform(test_data[col])

X_test = test_data.drop(columns=[41, 42])
y_test = test_data[41]

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))