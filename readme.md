Fake News Detection System
A machine learning application that classifies news articles as real or fake using a fine-tuned BERT model.

1. Install Dependencies
Install the required libraries using pip:
pip install streamlit torch transformers pillow nest_asyncio and any other libraries present that you may not have.

2. Update Model Path
Update the model path in app.py to match where you store the BERT model files:
MODEL_PATH = r"C:\Users\YOUR_USERNAME\OneDrive\Desktop\Project Dissertation\bert_results\bertModel\BERT"

Running the Application
Run the app with Streamlit:
streamlit run app.py

Usage:

Paste a news article in the text box.

Click "Check News."

The system will show:

"Real" for real news

"Fake" for fake news

Confidence Score – how confident the model is

Disclaimer
This system is for informational purposes only. Predictions are based on machin