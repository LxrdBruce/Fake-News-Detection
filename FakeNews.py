import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import re
from sklearn.dummy import DummyClassifier

# Loading the training dataset
df = pd.read_csv('train.csv')

print(df.head())
print(df.info())

df = df.dropna(subset=['text'])

df['title'] = df['title'].fillna('No Title')
df['author'] = df['author'].fillna('Unknown')

print(df.info())

nltk.download('punkt')

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', '', text) 
    return text.lower().strip()

df['processed_text'] = df['text'].apply(clean_text)

# Filtering out stopwords
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),  
    max_df=0.9, 
    min_df=2 
)

# Class Balance Check
sns.countplot(x='label', data=df)
plt.title('Class Balance: Fake (1) vs Real (0)')
plt.show()

# Ensuring 'label' column exists
if 'label' not in df.columns:
    raise ValueError("Label column not found in the dataset.")

# Split into training and testing sets
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)


print("Training size:", X_train.shape, "Test size:", X_test.shape)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_pred = dummy_clf.predict(X_test)

print("Baseline Dummy Classifier Accuracy:", accuracy_score(y_test, dummy_pred))

# Initialise SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Trainining the SVM model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))


labels = list(set(y_test)) 

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for SVM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best SVM Parameters:", grid_search.best_params_)
print("Best SVM Accuracy on Training Set:", grid_search.best_score_)

# Evaluate the best model
best_svm_model = grid_search.best_estimator_
y_pred_best_svm = best_svm_model.predict(X_test)
print("\nBest SVM Classification Report:\n", classification_report(y_test, y_pred_best_svm))

# Confusion Matrix for the best SVM
cm_best_svm = confusion_matrix(y_test, y_pred_best_svm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_best_svm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for Best SVM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



#  train Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# predict on the test set
y_pred = nb_model.predict(X_test)

# evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for Naïve Bayes Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Comparison of Models:")
print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
plt.savefig("svm_confusion_matrix.png")