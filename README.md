# Binary Text Classification using Logistic Regression

This project demonstrates how to perform binary text classification using Natural Language Processing (NLP) with Logistic Regression. The code classifies text documents from the 20 Newsgroups dataset into two categories: 'rec.sport.baseball' and 'sci.med'. It uses TF-IDF vectorization to preprocess the text data and Logistic Regression as the classification model.
Requirements

📌 To run this project, you need to have Python installed along with the necessary libraries. You can install them using pip:
 
    pip install numpy pandas scikit-learn

🚀 Required Libraries:

- numpy (for numerical operations)
- pandas (for data manipulation)
- scikit-learn (for machine learning, including models and text preprocessing)

🖥️ Dataset

This project uses the 20 Newsgroups dataset available from scikit-learn. The dataset consists of newsgroup posts categorized into 20 different topics. For this binary classification task, we have filtered the dataset to use only two categories:

- 'rec.sport.baseball'
- 'sci.med'

📊  How It Works

The code follows these steps:
1. Data Loading

We load the 20 Newsgroups dataset using fetch_20newsgroups from sklearn.datasets, specifically selecting the two categories ('rec.sport.baseball' and 'sci.med') for binary classification.

2. Data Splitting

The dataset is split into training and testing sets. 80% of the data is used for training the model, and the remaining 20% is used for evaluating the model.
3. Text Preprocessing with TF-IDF

   TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numeric features. The TfidfVectorizer from scikit-learn handles this preprocessing.
   The text data is transformed into a TF-IDF matrix, where each document is represented by a vector of numeric values (weights).

4. Model Training with Logistic Regression

    A Logistic Regression model is trained using the processed training data. Logistic Regression is commonly used for binary classification tasks due to its simplicity and effectiveness.

5. Prediction & Evaluation

    The trained model is evaluated on the test data using accuracy and classification report metrics (which include precision, recall, and F1-score).
    A sample text document is classified using the trained model to demonstrate prediction.

6. Sample Prediction

    A sample document, "I love playing baseball during the summer.", is classified by the model, and the predicted category (either 0 for rec.sport.baseball or 1 for sci.med) is printed.

🛠️ How to Run the Code
1. Clone the Repository
Clone this repository to your local machine:
 
       git clone https://github.com/your-username/binary-text-classification.git

2. Navigate to the Project Folder

       cd binary-text-classification

3. Install Dependencies
Install the required dependencies by running:

       pip install -r requirements.txt

If you don't have requirements.txt, you can manually install the necessary libraries:

       pip install numpy pandas scikit-learn

4. Run the Python Script
Execute the Python script to train the model and see the results:

       python binary_text_classification.py

