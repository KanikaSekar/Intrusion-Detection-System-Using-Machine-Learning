import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
# Load train data
train_data = pd.read_csv("dataset/KDDTrain+.txt", header=None)

# Binary labels
train_data[41] = train_data[41].apply(
    lambda x: 0 if x == "normal" else 1
)

# One-hot encoding
train_data = pd.get_dummies(
    train_data,
    columns=[1, 2, 3]
)
train_data.columns = train_data.columns.astype(str)



# Split features and labels
X_train = train_data.drop(columns=["41", "42"])
y_train = train_data["41"]

# Train model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Load test data
test_data = pd.read_csv("dataset/KDDTest+.txt", header=None)

test_data[41] = test_data[41].apply(
    lambda x: 0 if x == "normal" else 1
)

test_data = pd.get_dummies(
    test_data,
    columns=[1, 2, 3]
)
test_data.columns = test_data.columns.astype(str)
# Match train/test columns
X_test = test_data.drop(columns=["41", "42"])
y_test = test_data["41"]

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", round(
    accuracy_score(y_test, y_pred) * 100, 2
), "%")
print("ROC-AUC:", round(auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))