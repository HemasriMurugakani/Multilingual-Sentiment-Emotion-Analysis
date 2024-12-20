# Multilingual-Sentiment-Emotion-Analysis
A web-based tool developed to provide sentiment and emotion insights for text in multiple languages!

# Project Overview
This project uses a combination of **VADER Sentiment Analysis**, **Hugging Face's BERT-based Transformers**, and **Streamlit** for creating an interactive web-based application that provides the following features:

- **Multilingual Support**: Supports multiple languages and accurately detects them using a custom-trained language detection model.
- **VADER Sentiment Analysis**: Performs polarity scoring for basic sentiment analysis.
- **Emotion Classification**: Classifies emotions like joy, anger, fear, etc., using a transformer model.
- **Word Cloud Generation**: Visualizes the most frequent words in the input text.
- **Data Export**: Easily export the analysis results to **JSON** or **CSV** for reporting.

## Technologies Used
- **Streamlit**: For creating the interactive user interface.
- **NLTK (VADER Sentiment Analysis)**: For sentiment analysis and text preprocessing.
- **Hugging Face Transformers**: For advanced transformer-based models for sentiment and emotion classification.
- **Matplotlib & Seaborn**: For generating visualizations such as sentiment distribution and emotion analysis.
- **WordCloud**: For generating a visual representation of the most frequent words in the text.
- **Pandas & JSON**: For data handling and exporting analysis results.

## Contributing
Feel free to fork the repository, open issues, and submit pull requests for improvements! Contributions are welcome, whether it's bug fixes, new features, or documentation improvements.

## Acknowledgments
NLTK: For providing the VADER Sentiment Analysis tool.
Hugging Face: For the amazing transformer models used in sentiment and emotion detection.
Streamlit: For the user-friendly and interactive app framework.
