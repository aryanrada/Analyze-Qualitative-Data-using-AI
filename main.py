import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bertopic import BERTopic
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import organizer

st.sidebar.title("Analyze Qualitative Data using AI")

def sentiment_analyzer_scores(text):
    analyzer = SentimentIntensityAnalyzer()
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

    model = st.sidebar.radio("Select Model", ("VADAR", "BERTopic", "Spacy NER"))

    if st.sidebar.button("Show Analysis"):
        data = organizer.fetch_data(option,data)

        if model == 'VADAR':
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

        elif model == "BERTopic":
            #data = organizer.bertopic(data)
            st.title("BERTopic")
            topic_model = BERTopic()
            topics, probabilities = topic_model.fit_transform(data["Title"])
            data['Title_Topic'] = topics
            data['Title_Probability'] = probabilities
            topic_data = topic_model.get_topic_info()
            topic_hierarchy = topic_model.visualize_hierarchy()
            topic_visual = topic_model.visualize_topics()
            st.header("Analyzed data")
            st.write(topic_data)
            st.header("Visual Representation")
            st.write(topic_visual)
            st.write(topic_hierarchy)

        elif model == "Spacy NER":
            model = spacy.load("en_core_web_sm")

            def ner(text):
                doc = model(text)
                entities = []
                for ent in doc.ents:
                    entities.append((ent.text, ent.label_))
                return entities
            
            data['entities'] = data['Title'].apply(ner)

            ner_data = []
            for index, row in data.iterrows():
                text = row['Title']
                for entity, entity_type in row['entities']:
                    ner_data.append([text, entity, entity_type])

            ner_data = pd.DataFrame(ner_data, columns=['text', 'entity', 'entity_type'])

            st.title("Spacy NER")
            st.header("Analyzed data")
            st.write(ner_data)

            st.header("Graphical Representation")
            entity_count = ner_data["entity_type"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(entity_count)
            st.pyplot(fig)

        
