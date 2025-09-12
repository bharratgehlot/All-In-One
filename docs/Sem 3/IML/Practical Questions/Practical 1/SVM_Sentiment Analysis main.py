import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Download the movie_reviews dataset if you haven't already
nltk.download("movie_reviews")

# Load the movie_reviews dataset
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
labels = [1 if fileid.split('/')[0] == "pos" else 0 for fileid in movie_reviews.fileids()]

print(type(reviews))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train an SVM classifier
clf = SVC(kernel='linear')  # You can try different kernels (e.g., 'linear', 'rbf')
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report for more details
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)