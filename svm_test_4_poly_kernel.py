import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file
data = pd.read_csv('Updated_dataset_with_full_text.csv')

# Display the first few rows to understand the structure of the data
print(data.head())

# Use the 'Text' column for the tweets and 'Biased' column for the labels
texts = data['Text']
labels = data['Biased']

# Convert the text data into numerical values using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create an SVM classifier with a polynomial kernel
svm = SVC(kernel='poly', degree=3, coef0=1, C=1)  # You can experiment with different degrees and coef0 values

# Train the model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Evaluate the model's performance
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Classification report provides detailed metrics like precision, recall, and F1-score for each category
report = classification_report(y_test, y_pred, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])
print(report)
