import librosa
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Feature Extraction Function
def extract_feature(file_name):

    audio, sample_rate = librosa.load(
        file_name,
        res_type='kaiser_fast'
    )

    mfccs = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    return mfccs


# Dataset Folder
dataset_path = r"C:\Users\shara\OneDrive\ドキュメント\Machine Learning\Maths for ml\CodeAlpha_EmotionRecognition\dataset"


# Empty Lists
X = []
Y = []


# Read Audio Files
for file in os.listdir(dataset_path):

    if file.endswith(".wav"):

        # Emotion Name from filename
        # Example: happy.wav → happy
        emotion = file.split(".")[0]

        file_path = os.path.join(dataset_path, file)

        feature = extract_feature(file_path)

        X.append(feature)

        Y.append(emotion)


# Convert to numpy arrays
X = np.array(X)

# Encode labels
encoder = LabelEncoder()

Y = encoder.fit_transform(Y)


# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# Create Model
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500
)


# Train Model
model.fit(x_train, y_train)


# Predictions
y_pred = model.predict(x_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Test Custom Audio
print("\n----- Emotion Prediction -----")

test_file = input("Enter test audio filename: ")

feature = extract_feature(test_file)

feature = feature.reshape(1, -1)

prediction = model.predict(feature)

emotion = encoder.inverse_transform(prediction)

print("\nPredicted Emotion:", emotion[0])