import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import organizer
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
    data = data.drop(columns=['Unnamed: 0', 'is_Original', 'Flair', 'URL'])
    data["Title"].fillna("Null", inplace = True)
    data["Body"].fillna("Null", inplace = True)
    data["Comments"].fillna("Null", inplace = True)

    subreddit = data['Subreddit'].unique().tolist()
    subreddit.sort()
    subreddit.insert(0,"Overall")
    option = st.sidebar.selectbox("Select Subreddit", subreddit)

    if st.sidebar.button("Show Analysis"):
        data = organizer.fetch_data(option,data)

        data['Title_Sentiment'] = data['Title'].apply(sentiment_analyzer_scores)
        
        st.title("Sentiment Analysis")
        
        st.header("Overall Sentiment Analysis")
        fig, ax = plt.subplots()
        ax.pie(data['Title_Sentiment'].value_counts(), labels = data['Title_Sentiment'].value_counts().index, autopct='%1.1f%%')
        st.pyplot(fig)
        
        st.header("Sentiment Analysis by Subreddit")
        fig, ax = plt.subplots()
        ax = sns.countplot(x='Subreddit', hue='Title_Sentiment', data=data)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.header("Sentiment counts by Subreddit")
        sentiment_counts = data['Title_Sentiment'].groupby(data['Subreddit']).value_counts().unstack().fillna(0)
        sentiment_counts_df = sentiment_counts.reset_index()
        sentiment_counts_df.columns.name = None
        st.write(sentiment_counts_df)
