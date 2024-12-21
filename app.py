import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import json
import pandas as pd
from langdetect import detect

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

sia = SentimentIntensityAnalyzer()
transformer_model_en = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
emotion_model = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
lang_model = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


def preprocess_text(text):
    text = text.lower()
    return text

def detect_language(text):
    result = lang_model(text)
    lang = result[0]['label']
    confidence = result[0]['score']
    return lang, confidence

def analyze_sentiment(text):
    text = preprocess_text(text)
    lang, confidence = detect_language(text)

    if confidence < 0.9:
        st.warning("Language detection confidence is low. Results may be inaccurate.")
    
    ss = sia.polarity_scores(text)
    vader_sentiment = "Neutral"
    if ss['compound'] >= 0.05:
        vader_sentiment = "Positive"
    elif ss['compound'] <= -0.05:
        vader_sentiment = "Negative"

    transformer_result = transformer_model_en(text)
    transformer_sentiment = transformer_result[0]['label']

    emotion_result = emotion_model(text)
    emotion_scores = {res['label']: res['score'] for res in emotion_result}

    return {
        'Language': lang,
        'Confidence': confidence,
        'VADER Sentiment': vader_sentiment,
        'Transformer Sentiment': transformer_sentiment,
        'Emotion Scores': emotion_scores,
        'Detailed Scores': ss
    }
st.set_page_config(page_title="Multilingual Sentiment Analysis", layout="wide")

st.title("\U0001F31F Multilingual Sentiment & Emotion Analysis \U0001F30D")
st.caption("Powered by NLTK, Hugging Face Transformers, and Streamlit")

with st.sidebar:
    st.header("Settings")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
    show_wordcloud = st.checkbox("Show Word Cloud", value=True)
    export_results = st.checkbox("Enable Export Options", value=True)
    
    st.subheader("About the Developer")
    st.write("""
    Developed by **M Hemasri Murugakani**
    """)
    st.write("Explore and try out different texts for analysis.")

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

text = st.text_area("Enter your text here (any language):", placeholder="Type something...")

if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(text)

        st.subheader("Results:")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Detected Language:** {result['Language']} (Confidence: {result['Confidence']:.2f})")
            st.write(f"**VADER Sentiment:** {result['VADER Sentiment']}")
            st.write(f"**Transformer Sentiment:** {result['Transformer Sentiment']}")

        with col2:
            st.write("**Text Insights:**")
            st.write(f"**Word Count:** {len(text.split())}")
            st.write(f"**Character Count:** {len(text)}")

        st.subheader("Visualizations:")
        ss = result['Detailed Scores']
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_scores = [ss['pos'], ss['neu'], ss['neg']]

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Sentiment Breakdown (VADER)")
            fig, ax = plt.subplots()
            ax.bar(sentiment_labels, sentiment_scores, color=['green', 'gray', 'red'])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Score")
            ax.set_title("VADER Sentiment Analysis")
            st.pyplot(fig)

        with col4:
            st.markdown("### Emotion Distribution")
            emotion_scores = result['Emotion Scores']
            fig, ax = plt.subplots()
            ax.bar(emotion_scores.keys(), emotion_scores.values(), color=sns.color_palette("Set2"))
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Score")
            ax.set_title("Emotion Scores")
            st.pyplot(fig)

        if show_wordcloud:
            st.subheader("Word Cloud")
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            words = " ".join([word for word in text.split() if word not in stop_words])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        if export_results:
            st.subheader("Export Options")
            export_data = {
                "Language": result['Language'],
                "VADER Sentiment": result['VADER Sentiment'],
                "Transformer Sentiment": result['Transformer Sentiment'],
                "Emotion Scores": result['Emotion Scores'],
                "Detailed Scores": result['Detailed Scores']
            }
            export_json = json.dumps(export_data, indent=4)
            export_df = pd.DataFrame([export_data])

            st.download_button("Download as JSON", data=export_json, file_name="analysis_results.json", mime="application/json")
            st.download_button("Download as CSV", data=export_df.to_csv(index=False), file_name="analysis_results.csv", mime="text/csv")
    else:
        st.error("Please enter some text to analyze.")
