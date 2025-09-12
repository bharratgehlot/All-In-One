import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load your network traffic dataset (replace 'network_traffic.csv' with your dataset)
data = pd.read_csv('network_traffic.csv')

# Separate the feature columns from the target column (assuming 'label' indicates anomalies)
X = data.drop(['label','date'], axis=1)  # Features
y = data['label']  # Anomaly labels (1 for anomalies, -1 for normal)

# Standardize the feature data (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the One-Class SVM model
svm_model = OneClassSVM(kernel='rbf', nu=0.05)  # Adjust hyperparameters as needed
svm_model.fit(X_scaled)

# Predict anomalies in the dataset
y_pred = svm_model.predict(X_scaled)

# Evaluate the model's performance
classification_report_str = classification_report(y, y_pred)
print("Classification Report:\n", classification_report_str)