import re
import asyncio
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def clean_text(text):
    """Preprocess text similar to training script"""
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', '', text) 
    return text.lower().strip()

# Loading BERT model and tokeniser
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = r"C:\Users\ezank\OneDrive\Desktop\Project Dissertation\bert_results\bertModel\BERT"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=2, 
    ignore_mismatched_sizes=True
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Title and description
st.title("Fake News Detection System")
st.write("Enter a news article below to check if it's real or fake:")

# Text box
article = st.text_area("Paste the news article here", height=300)

if st.button("Check News"):
    if article.strip() != "":
        cleaned_article = clean_text(article)
        
        # Preprocessing
        inputs = tokenizer(
            cleaned_article, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Result
        label = "Real" if prediction == 0 else "Fake"
        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        
        if confidence < 0.6:
            st.warning("The confidence is relatively low. Consider verifying the source.")

    else:
        st.warning("Please enter a news article.")

st.markdown("---")
st.markdown("**Fake News Detection Project by Ezan Khan**")