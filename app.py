import re
import asyncio
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
from PIL import Image
import base64
import os



def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', '', text) 
    return text.lower().strip()

# Load BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = r"C:\Users\ezank\OneDrive\Desktop\Project Dissertation\bert_results\bertModel\BERT"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    num_labels=2, 
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Page Config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

logo_path = "logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

        st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px; margin-top: 40px;">
            <img src="data:image/png;base64,{encoded}" width="250" />
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='text-align: center; color: grey;'>[Optional Logo Not Found]</p>", 
        unsafe_allow_html=True
    )


# Title
st.markdown(
    "<h1 style='text-align: center; color: black;'>Fake News Detection System</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 200px;
            max-width: 250px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Text box
article = st.text_area("Paste the news article here", height=300)

# Prediction
if st.button("Check News"):
    if article.strip() != "":
        with st.spinner("Analyzing the news..."):
            cleaned_article = clean_text(article)

            if len(cleaned_article.split()) < 5:
                st.warning("⚠️ The article is too short to analyze.")
            elif len(cleaned_article.split()) > 1000:
                st.warning("⚠️ The article is too long — consider summarizing it.")
            else:
                # Preprocessing
                inputs = tokenizer(
                    cleaned_article, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=256
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = round(probabilities[0][prediction].item() * 100, 2)  # Round to 2 decimal places

                # Result
                label = "Real" if prediction == 0 else "Fake"
                if prediction == 0:
                    st.success(f"✅ Prediction: {label} (Confidence: {confidence}%)")
                else:
                    st.error(f"❌ Prediction: {label} (Confidence: {confidence}%)")

                
                if confidence < 60:
                    st.warning("⚠️ The confidence is relatively low. Consider verifying the source.")

                st.markdown("<br>", unsafe_allow_html=True)    
                st.info(
                    "🔍 This is a machine learning model. While it performs well in most cases, it may occasionally "
                    "misclassify articles, even with high confidence. Please verify from trusted sources if in doubt."
                )
                
                
                st.write(f"🕒 Prediction generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("⚠️ Please enter a news article.")

with st.sidebar:
    st.header("About the Model")
    st.write(
        "This fake news detection system uses a fine-tuned BERT model. "
        "It classifies news articles as real or fake based on the language patterns "
        "and structure of the text."
    )
    st.header("How to Use:")
    st.write(
        "- Paste the full news article into the text box.\n"
        "- Click **Check News**.\n"
        "- The system will return a prediction with a confidence score.\n"
        "- Low confidence means the result may not be reliable."
    )

    st.write("ℹ️ **Disclaimer:** This system is for informational purposes only. Predictions are based on machine learning models and may not always be accurate. Confidence scores reflect the model's certainty based on patterns in the training data, but errors are still possible.")

#Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Fake News Detection Project by Ezan Khan</p>", 
    unsafe_allow_html=True
)