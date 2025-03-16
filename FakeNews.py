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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.svm import SVC
import re
import os
os.environ["WANDB_DISABLED"] = "true"
from sklearn.dummy import DummyClassifier
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import random

# Random seed for consistency (BERT)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




# Loading the training dataset
df = pd.read_csv('train.csv')

print(df.head())
print(df.info())

df = df.dropna(subset=['text'])

df['title'] = df['title'].fillna('No Title')
df['author'] = df['author'].fillna('Unknown')

print(df.info())

nltk.download('punkt')

# Load BERT tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def plot_roc_pr_curves(model_name, y_test, y_probs):
    """Plots ROC and Precision-Recall curves for a given model."""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="red", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")

    plt.show()


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

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label.clone().detach()

        }


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

tokenizer_lstm = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer_lstm.fit_on_texts(df['processed_text'])

# Convert text to sequences
X_train_seq = [seq if len(seq) > 0 else [0] for seq in tokenizer_lstm.texts_to_sequences(X_train)]
X_test_seq = [seq if len(seq) > 0 else [0] for seq in tokenizer_lstm.texts_to_sequences(X_test)]


# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Prepare Data for BERT
train_labels = torch.tensor(y_train.tolist(), dtype=torch.long)
test_labels = torch.tensor(y_test.tolist(), dtype=torch.long)

train_dataset = FakeNewsDataset(X_train.tolist(), train_labels, tokenizer)
test_dataset = FakeNewsDataset(X_test.tolist(), test_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}



training_args = TrainingArguments(
    output_dir="./bert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./bert_logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3, 
    weight_decay=0.01,
    fp16=True
    
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# Train BERT
print("\nTraining BERT Model...")
trainer.train()

print("\nEvaluating BERT Model...")
bert_eval_results = trainer.evaluate()
print("BERT Evaluation Results:", bert_eval_results)

def plot_bert_roc_pr_curves(trainer, dataset, model_name="BERT"):
    predictions = trainer.predict(dataset)
    y_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].cpu().numpy()
    y_test = np.array([example["labels"].item() for example in dataset])


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="red", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")

    plt.show()

# Call the function to plot BERT's ROC and PR curves
plot_bert_roc_pr_curves(trainer, test_dataset)




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


labels = ["Real", "Fake"]


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
y_probs_svm = best_svm_model.predict_proba(X_test_tfidf)[:, 1]
print("\nBest SVM Classification Report:\n", classification_report(y_test, y_pred_best_svm))
plot_roc_pr_curves("SVM", y_test, y_probs_svm)

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

y_probs_lstm = model_lstm.predict(X_test_pad).flatten()

plot_roc_pr_curves("LSTM", y_test, y_probs_lstm)

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
y_probs = nb_model.predict_proba(X_test_tfidf)[:, 1]

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

plot_roc_pr_curves("Naïve Bayes", y_test, y_probs)

print("Comparison of Models:")
print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(f"LSTM Accuracy: {lstm_acc}")
bert_accuracy = bert_eval_results.get("eval_accuracy", "N/A")
print(f"BERT Accuracy: {bert_accuracy}")
plt.savefig("svm_confusion_matrix.png")