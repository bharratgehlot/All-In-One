# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load your dataset (replace 'your_dataset.csv' with your data file)
data = pd.read_csv('Telco-Customer-Churn.csv')

# Step 3: Preprocess the data
# You may need to perform data cleaning, encoding, and feature engineering here

# Example: Encoding categorical variables (if needed)
data = pd.get_dummies(data, columns=['Contract', 'PaymentMethod'])

# Example: Standardize numerical features
#scaler = StandardScaler()
#data[['MonthlyCharges'], ['TotalCharges']] = scaler.fit_transform(data[['MonthlyCharges', 'TotalCharges']])

# Step 4: Split the data into training and testing sets
X = data[['Contract']]  # Features
y = data['churn']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model's performance
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

# Generate a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Step 7: Make predictions on new data
# You can use the trained model to make predictions on new customer data

# Example:
new_customer_data = pd.DataFrame({

    'Contract': ['One year'],  # Encode categorical variables as needed
    'PaymentMethod': ['Mailed check']
})

new_prediction = model.predict(new_customer_data)
print(f'New Customer Churn Prediction: {new_prediction[0]}')