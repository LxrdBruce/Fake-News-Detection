import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the training dataset
df = pd.read_csv('train.csv')

print(df.head())
print(df.info())

df = df.dropna(subset=['text'])

df['title'] = df['title'].fillna('No Title')
df['author'] = df['author'].fillna('Unknown')

print(df.info())

nltk.download('punkt')

# Tokenising the 'text' column and add a new column 'tokens'
df['tokens'] = df['text'].apply(word_tokenize)

print(df[['text', 'tokens']].head())

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Filter out stop words
df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
print(df[['text', 'tokens']].head())

# Convert the tokens back to full strings
df['processed_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

# Initialising TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_text'])

# Checking the shape of the TF-IDF matrix
print("TF-IDF matrix shape:", X.shape)

# Class Balance Check
sns.countplot(x='label', data=df)
plt.title('Class Balance: Fake (1) vs Real (0)')
plt.show()

# Token Count Distribution
df['token_count'] = df['tokens'].apply(len)
sns.histplot(df['token_count'], bins=30)
plt.title('Distribution of Token Counts per Article')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.show()

# Flatten list of all tokens there
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words)

common_words = word_freq.most_common(20)
words, frequencies = zip(*common_words)

plt.figure(figsize=(10, 6))
sns.barplot(x=frequencies, y=words)
plt.title('Top 20 Most Common Words')
plt.xlabel('Frequency')
plt.show()

# Ensuring 'label' column exists
if 'label' not in df.columns:
    raise ValueError("Label column not found in the dataset.")

# Split into training and testing sets
y = df['label']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape, "Test size:", X_test.shape)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title("Confusion Matrix for Naïve Bayes Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
