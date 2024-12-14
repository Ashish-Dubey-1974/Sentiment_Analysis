# -*- coding: utf-8 -*-
try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' library is not installed. Install it using 'pip install streamlit'.")

from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
from PyPDF2 import PdfReader
import docx

def convert_to_df(sentiment):
    return pd.DataFrame({
        'metric': ['polarity', 'subjectivity'],
        'value': [sentiment.polarity, sentiment.subjectivity]
    })

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []

    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append((i, res))
        elif res <= -0.1:
            neg_list.append((i, res))
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

def read_file(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    elif file.type == "text/plain":
        stringio = StringIO(file.getvalue().decode("utf-8"))
        return stringio.read()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "".join([para.text for para in doc.paragraphs])
    else:
        return None

def main():
    st.title("Sentiment Analysis using NLP")
    st.subheader("Simple Sequence Classification")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        
        input_mode = st.radio("Input Mode", ("Write Text", "Upload File"))
        raw_text = ""

        if input_mode == "Write Text":
            raw_text = st.text_area("Enter Text Here")
        else:
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
            if uploaded_file is not None:
                raw_text = read_file(uploaded_file)
                if raw_text:
                    st.text_area("File Content", raw_text, height=200)

        if raw_text:
            submit_button = st.button(label='Analyze')
            values = st.slider("Select a range", -1.0, 1.0, (-0.05, +0.05))
            col1, col2 = st.columns(2)

            if submit_button:
                with col1:
                    st.info("Results")
                    sentiment = TextBlob(raw_text).sentiment
                    st.write(sentiment)

                    if sentiment.polarity > values[1]:
                        st.markdown("Sentiment: Positive 😊")
                    elif sentiment.polarity < values[0]:
                        st.markdown("Sentiment: Negative 😠")
                    else:
                        st.markdown("Sentiment: Neutral 😐")

                    result_df = convert_to_df(sentiment)
                    st.dataframe(result_df)

                    c = alt.Chart(result_df).mark_bar().encode(
                        x='metric', y='value', color='metric'
                    )
                    st.altair_chart(c, use_container_width=True)

                with col2:
                    st.info("Token Sentiment")
                    token_sentiments = analyze_token_sentiment(raw_text)
                    st.write(token_sentiments)
    
    else:
        st.subheader("About")
        st.write("This app analyzes sentiment using TextBlob and VADER.")

if __name__ == '__main__':
    main()
