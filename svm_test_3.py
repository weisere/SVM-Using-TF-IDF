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

# Create an SVM classifier
svm = SVC(kernel='linear', class_weight='balanced', probability=True)  # Added class_weight='balanced'

# Train the model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred_proba = svm.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class (antisemitic)

# Adjust the decision threshold
threshold = 0.35  # Adjusting the threshold (you can experiment with different values)
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

# Evaluate the adjusted predictions
print(f'Adjusted Accuracy: {accuracy_score(y_test, y_pred_adjusted)}')
adjusted_report = classification_report(y_test, y_pred_adjusted, target_names=['(0) Non-antisemitic', '(1) Antisemitic'])
print(adjusted_report)
