#in pycharm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"D:\PROJECT\ML\drug200.csv")

# Check column names
print("Columns in dataset:", data.columns)

# Specify the target column
target_column = 'Drug'

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features
categorical_columns = ['Sex', 'BP', 'Cholesterol']
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Batch Gradient Descent Implementation
class BatchGradientDescentClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize weights and bias
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # One-hot encoding for the target
        y_encoded = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            y_encoded[i, label] = 1

        # Gradient Descent
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            probabilities = self._softmax(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_encoded))
            db = (1 / n_samples) * np.sum(probabilities - y_encoded, axis=0)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(linear_model)
        return np.argmax(probabilities, axis=1)

    def _softmax(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Initialize and train the Batch Gradient Descent model
model = BatchGradientDescentClassifier(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and display accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot performance metrics: Loss vs. Epochs
# Placeholder for loss history if you want to track loss over epochs
loss_history = []
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs for Batch Gradient Descent")
plt.show()
