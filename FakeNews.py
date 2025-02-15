import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
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
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam


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

#Split into training and testing sets
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

#Convert text to sequences for LSTM
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 500 

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['processed_text'])

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

X_train_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'][X_train.index])
X_test_tfidf = tfidf_vectorizer.transform(df['processed_text'][X_test.index])

print("TF-IDF Training size:", X_train_tfidf.shape, "Test size:", X_test_tfidf.shape)
print("LSTM Training size:", X_train_pad.shape, "Test size:", X_test_pad.shape)


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_tfidf, y_train)
dummy_pred = dummy_clf.predict(X_test_tfidf)

print("Baseline Dummy Classifier Accuracy:", accuracy_score(y_test, dummy_pred))

# Initialise SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Trainining the SVM model
svm_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluate the SVM model
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

print(f"Length of y_test: {len(y_test)}")


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
param_grid = {'C': [1], 'kernel': ['linear']}
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1, 
    verbose=3 
)

grid_search.fit(X_train_tfidf, y_train)


# Best parameters and score
print("Best SVM Parameters:", grid_search.best_params_)
print("Best SVM Accuracy on Training Set:", grid_search.best_score_)

# Evaluate the best model
best_svm_model = grid_search.best_estimator_
y_pred_best_svm = best_svm_model.predict(X_test_tfidf)
print("\nBest SVM Classification Report:\n", classification_report(y_test, y_pred_best_svm))

# Confusion Matrix for the best SVM
cm_best_svm = confusion_matrix(y_test, y_pred_best_svm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_best_svm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for Best SVM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# LSTM Model
LSTM_UNITS = 128

model_lstm = Sequential([
    Embedding(MAX_NB_WORDS, 100, input_length=MAX_SEQUENCE_LENGTH),
    SpatialDropout1D(0.2),
    LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') 
])

# Compile LSTM Model
model_lstm.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train LSTM Model
print("\nTraining LSTM Model...")
history_lstm = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test), verbose=1)

# Evaluate LSTM Model
lstm_loss, lstm_acc = model_lstm.evaluate(X_test_pad, y_test)
print("\nLSTM Accuracy:", lstm_acc)

# LSTM Classification Report
y_pred_lstm = (model_lstm.predict(X_test_pad) > 0.5).astype("int32")

print("\nLSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))

# Confusion Matrix for LSTM
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for LSTM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



#  train Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# predict on the test set
y_pred = nb_model.predict(X_test_tfidf)

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
print(f"LSTM Accuracy: {lstm_acc}")
plt.savefig("svm_confusion_matrix.png")