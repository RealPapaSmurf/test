from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Import required libraries

# Generate synthetic dataset for classification
# 100 samples, 4 features, 2 classes
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Calculate and print accuracy
accuracy = np.mean(predictions == y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Example of predicting a new sample
new_sample = X_test[0].reshape(1, -1)
prediction = model.predict(new_sample)
print(f"Prediction for new sample: {prediction[0]}")