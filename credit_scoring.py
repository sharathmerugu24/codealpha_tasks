# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Load dataset
data = pd.read_csv(r"C:\Users\shara\OneDrive\ドキュメント\Machine Learning\Maths for ml\CodeAlpha_CreditScoring\german_credit_data.csv")

print("First 5 Rows:")
print(data.head())


# Drop unnecessary column
data.drop("Unnamed: 0", axis=1, inplace=True)


# Fill missing values
data["Saving accounts"].fillna("unknown", inplace=True)
data["Checking account"].fillna("unknown", inplace=True)


# Create target column
# 0 = Creditworthy
# 1 = Not Creditworthy

data["Risk"] = np.where(data["Credit amount"] > 5000, 1, 0)


# Encode categorical columns
label_encoder = LabelEncoder()

categorical_columns = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])


# Features and target
X = data.drop("Risk", axis=1)
y = data["Risk"]


# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LogisticRegression()


# Train model
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
print("\n----- Predict New Customer -----")

age = int(input("Enter Age: "))

sex = input("Enter Sex (male/female): ")

job = int(input("Enter Job Level (0-3): "))

housing = input("Enter Housing (own/rent/free): ")

saving_accounts = input(
    "Enter Saving Account Status (little/moderate/rich/quite rich/unknown): "
)

checking_account = input(
    "Enter Checking Account Status (little/moderate/rich/unknown): "
)

credit_amount = int(input("Enter Credit Amount: "))

duration = int(input("Enter Duration: "))

purpose = input(
    "Enter Purpose (car/radio-TV/education/furniture-equipment/business/etc): "
)


# Encode user inputs
sex = label_encoder.fit_transform([sex])[0]
housing = label_encoder.fit_transform([housing])[0]
saving_accounts = label_encoder.fit_transform([saving_accounts])[0]
checking_account = label_encoder.fit_transform([checking_account])[0]
purpose = label_encoder.fit_transform([purpose])[0]


# Prepare input
new_data = [[
    age,
    sex,
    job,
    housing,
    saving_accounts,
    checking_account,
    credit_amount,
    duration,
    purpose
]]

# Scale input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)

if prediction[0] == 0:
    print("\nCustomer is Creditworthy")
else:
    print("\nCustomer is NOT Creditworthy")