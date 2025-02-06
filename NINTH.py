#IN PYCHARM

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r"D:\PROJECT\ML\ionosphere.csv", header=None)

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode the target labels to binary (0, 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the feature data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize prototypes (two classes: 0 and 1)
prototypes = {
    0: np.mean(X_train[y_train == 0], axis=0),
    1: np.mean(X_train[y_train == 1], axis=0)
}

# Define learning rate and epochs
learning_rate = 0.1
epochs = 10

# LVQ training
for epoch in range(epochs):
    for i in range(len(X_train)):
        # Get current sample and its class
        sample = X_train[i]
        label = y_train[i]

        # Find the nearest prototype
        nearest_class = min(prototypes.keys(), key=lambda c: np.linalg.norm(sample - prototypes[c]))

        # Update the prototype
        if nearest_class == label:
            prototypes[nearest_class] += learning_rate * (sample - prototypes[nearest_class])
        else:
            prototypes[nearest_class] -= learning_rate * (sample - prototypes[nearest_class])

# LVQ prediction
y_pred = []
for sample in X_test:
    # Find the nearest prototype
    nearest_class = min(prototypes.keys(), key=lambda c: np.linalg.norm(sample - prototypes[c]))
    y_pred.append(nearest_class)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of"
      f" custom LVQ on Ionosphere dataset: {accuracy:.2f}")

# Comment on efficacy
if accuracy > 0.8:
    print(
        "The custom LVQ algorithm performed well on the Ionosphere dataset, indicating its efficacy in binary classification tasks.")
else:
    print(
        "The LVQ model's performance could be improved on this dataset, suggesting the need for parameter tuning or additional preprocessing.")
