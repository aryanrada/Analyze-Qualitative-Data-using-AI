import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.sidebar.title("Analyze Qualitative Data using AI")

analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05 :
        Sentiment = 'Positive'
    elif scores['compound'] <= -0.05 :
        Sentiment = 'Negative'
    else :
        Sentiment = 'Neutral'
    return Sentiment

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    #bytes_data = uploaded_file.getvalue()
    #data = data.decode("utf-8")
    #data = pd.DataFrame(data)
    data = data.drop(columns=['Unnamed: 0', 'is_Original', 'Flair', 'URL'])
    data["Title"].fillna("Null", inplace = True)
    data["Body"].fillna("Null", inplace = True)
    data["Comments"].fillna("Null", inplace = True)

    if st.sidebar.button("Show Analysis"):
        data['Title_Sentiment'] = data['Title'].apply(sentiment_analyzer_scores)
        
        st.title("Sentiment Analysis")
        fig, ax = plt.subplots()
        ax.pie(data['Title_Sentiment'].value_counts(), labels = data['Title_Sentiment'].value_counts().index, autopct='%1.1f%%')
        st.pyplot(fig)

