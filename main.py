import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from io import BytesIO
import organizer
import algorithms

torch.classes.__path__ = []

st.sidebar.title("Analyze Qualitative Data using AI")

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
            data['Title_Sentiment'] = data['Title'].apply(algorithms.sentiment_analyzer)
            st.title("Sentiment Analysis")
        
            st.header("Overall Sentiment Analysis")
            fig, ax = plt.subplots()
            ax.pie(data['Title_Sentiment'].value_counts(), labels = data['Title_Sentiment'].value_counts().index, autopct='%1.1f%%')
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            with open("chart.png", "wb") as f:
                f.write(buf.getvalue())

            with open("chart.png", "rb") as file:
                st.download_button(
                    label="Download Chart",
                    data=file,
                    file_name="chart.png",
                    mime="image/png",
                )
            
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

            csv = organizer.convert_df(sentiment_counts_df)
            st.download_button(
                label="Download data",
                data=csv,
                file_name="data.csv",
                mime="text/csv",
            )

        elif model == "BERTopic":
            st.title("BERTopic")
            topic_model, data = algorithms.bertopic(data)
            topic_data = topic_model.get_topic_info()
            topic_hierarchy = topic_model.visualize_hierarchy()
            topic_visual = topic_model.visualize_topics()
            st.header("Analyzed data")
            st.write(topic_data)

            csv = organizer.convert_df(topic_data)
            st.download_button(
                label="Download data",
                data=csv,
                file_name="data.csv",
                mime="text/csv",
            )

            st.header("Visual Representation")
            #st.write(topic_visual)

            fig, ax = plt.subplots()
            topic_visual.write_image(ax)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            with open("chart.png", "wb") as f:
                f.write(buf.getvalue())

            with open("chart.png", "rb") as file:
                st.download_button(
                    label="Download Chart",
                    data=file,
                    file_name="chart.png",
                    mime="image/png",
                )

            st.header("Topic Hierarchy")
            st.write(topic_hierarchy)

        elif model == "Spacy NER":
            st.title("Spacy NER")

            data = algorithms.spacy_ner(data)

            st.header("Analyzed data")
            st.write(data)

            csv = organizer.convert_df(data)
            st.download_button(
                label="Download data",
                data=csv,
                file_name="data.csv",
                mime="text/csv",
            )

            st.header("Graphical Representation")
            entity_count = pd.DataFrame(data["entity_type"].value_counts())
            #st.write(entity_count)
            fig, ax = plt.subplots()
            ax.bar(entity_count.index, entity_count.values.flatten())
            plt.xticks(rotation=90)
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            with open("chart.png", "wb") as f:
                f.write(buf.getvalue())

            with open("chart.png", "rb") as file:
                st.download_button(
                    label="Download Chart",
                    data=file,
                    file_name="chart.png",
                    mime="image/png",
                )

        
