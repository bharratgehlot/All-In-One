import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'credit_data.csv' with your dataset)
data = pd.read_csv('UCI_Credit_Card.csv')

# Preprocess the data (assuming you have already preprocessed your dataset)
# You may need to handle missing values, encode categorical variables, etc.

# Split the data into features and target variable
X = data.drop('default', axis=1)  # Features
y = data['default']  # Target variable

# Split the data into training and testing sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report for more details
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)