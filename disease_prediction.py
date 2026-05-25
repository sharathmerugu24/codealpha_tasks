import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load Dataset
data = pd.read_csv(r"diabetes.csv")


# Features and Target
X = data.drop("Outcome", axis=1)

y = data["Outcome"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train Model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Custom Prediction
print("\n----- Diabetes Prediction -----")

pregnancies = int(input("Pregnancies: "))
glucose = int(input("Glucose Level: "))
blood_pressure = int(input("Blood Pressure: "))
skin_thickness = int(input("Skin Thickness: "))
insulin = int(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))


new_data = [[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]]


prediction = model.predict(new_data)


if prediction[0] == 1:
    print("\nPatient likely has Diabetes")
else:
    print("\nPatient likely does NOT have Diabetes")
