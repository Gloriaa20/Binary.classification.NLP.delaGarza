# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups

# Step 1: Load the dataset
# For this example, we'll use the '20 newsgroups' dataset which is a collection of newsgroup documents.
# We will filter the dataset to use only two categories for binary classification (e.g., 'rec.sport.baseball' and 'sci.med').

newsgroups = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball', 'sci.med'])
X = newsgroups.data  # Text data (documents)
y = newsgroups.target  # Labels (0: rec.sport.baseball, 1: sci.med)

# Step 2: Split the data into training and testing sets
# We will use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocessing the text data
# We'll use TfidfVectorizer to convert the text documents into numeric vectors
# The TfidfVectorizer will create a matrix where each column represents a word in the vocabulary, and the values are the term frequency-inverse document frequency (TF-IDF) scores.

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
X_test_tfidf = vectorizer.transform(X_test)  # Only transform the test data

# Step 4: Train a binary classification model
# We will use Logistic Regression as our classification model. It's a simple and effective model for binary classification tasks.

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)  # Train the model on the training data

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Step 6: Evaluate the model
# We will evaluate the model using accuracy and classification report (precision, recall, f1-score)

print("Accuracy Score: ", accuracy_score(y_test, y_pred))  # Accuracy of the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Detailed performance metrics

# Step 7: Example of making a prediction on new, unseen text
# Let's try to classify a new sample document

new_sample = ["I love playing baseball during the summer."]
new_sample_tfidf = vectorizer.transform(new_sample)  # Transform the new sample text
prediction = model.predict(new_sample_tfidf)  # Predict the category

print("\nPredicted category for the new sample:")
print("0: rec.sport.baseball, 1: sci.med")
print(f"Prediction: {prediction[0]}")  # 0 -> rec.sport.baseball, 1 -> sci.med
