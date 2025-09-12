# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load your dataset (replace 'customer_data.csv' with your dataset)
data = pd.read_csv('Invistico_Airline.csv')

# Preprocess the data (handle missing values, encode categorical variables, etc.)
# Assuming you have already preprocessed the dataset


# Split the data into features and target variable
X = data.drop('satisfaction', axis=1)  # Features
y = data['satisfaction']  # Target variable


# Define which features are categorical and which are numerical
categorical_features = ['Gender', 'Customer Type', 'Type of Travel','Class']
numerical_features = ['Age', 'Flight Distance']

# Preprocessing for numerical data: imputation
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data: imputation and one-hot encoding
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),  ('model', model)])

# Split the data into training and testing sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train a Gradient Boosting ensemble model

clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report for more details
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)