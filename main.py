# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups

def load_and_preprocess_data():
    """
    Load the '20 Newsgroups' dataset, filter it for binary classification,
    and preprocess the text data using TF-IDF vectorization.
    """
    # Load the dataset and filter to only two categories
    newsgroups = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball', 'sci.med'])
    X = newsgroups.data  # Text data (documents)
    y = newsgroups.target  # Labels (0: rec.sport.baseball, 1: sci.med)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the text data using TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
    X_test_tfidf = vectorizer.transform(X_test)  # Only transform the test data

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_model(X_train_tfidf, y_train):
    """
    Train the Logistic Regression model using the provided training data.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)  # Fit the model on the training data
    return model

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the trained model using the test data, returning accuracy and classification report.
    """
    y_pred = model.predict(X_test_tfidf)  # Predict the labels for the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    report = classification_report(y_test, y_pred)  # Detailed classification report
    return accuracy, report

def predict_new_sample(model, vectorizer, new_sample):
    """
    Predict the category for a new text sample using the trained model.
    """
    new_sample_tfidf = vectorizer.transform(new_sample)  # Transform the new sample text
    prediction = model.predict(new_sample_tfidf)  # Predict the category
    return prediction[0]  # Return the predicted category

def main():
    """
    Main function to execute the binary classification pipeline:
    1. Load and preprocess the data
    2. Train the model
    3. Evaluate the model
    4. Make predictions on new text samples
    """
    # Step 1: Load and preprocess the data
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess_data()

    # Step 2: Train the model
    model = train_model(X_train_tfidf, y_train)

    # Step 3: Evaluate the model
    accuracy, report = evaluate_model(model, X_test_tfidf, y_test)
    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Step 4: Predict on a new text sample
    new_sample = ["I love playing baseball during the summer."]
    prediction = predict_new_sample(model, vectorizer, new_sample)
    print("\nPredicted category for the new sample:")
    print("0: rec.sport.baseball, 1: sci.med")
    print(f"Prediction: {prediction}")  # 0 -> rec.sport.baseball, 1 -> sci.med

# The script starts executing here
if __name__ == "__main__":
    main()  # Call the main function when the script is executed



