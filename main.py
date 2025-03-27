import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
import organizer
import algorithms

st.set_page_config(layout="wide")

page_bg_img = """
    <style>
    .stApp {
        background-image: url('https://t4.ftcdn.net/jpg/09/69/28/53/360_F_969285367_NuVZTFpgZnpMobKKOXi5ktg1lpCvwgFP.jpg');
        background-size: cover;
    }
    </style>
    """
st.markdown(page_bg_img, unsafe_allow_html=True)

#torch.hub.list('pytorch/vision', force_reload=True)
torch.classes.__path__ = []

info = st.sidebar.checkbox("About")
if info:
    st.title("Analyze Qualitative Data using AI")
    st.subheader("This project focuses on leveraging Artificial Intelligence to analyze qualitative data, providing insights into unstructured datasets such as customer reviews, survey responses, and social media comments. By employing advanced AI techniques, the project aims to simplify qualitative data analysis, enabling users to derive actionable insights efficiently.")
    st.subheader('''Files supported: :blue-background[CSV], :blue-background[PDF], :blue-background[DOCX], :blue-background[XLSX]''')
    st.subheader("Models available:")
    st.markdown(''' - :blue-background[VADAR] : VADER (Valence Aware Dictionary and sEntiment Reasoner) is designed to handle sentiments in social media text and informal language. Unlike traditional sentiment analysis methods, VADER is tailored to detect sentiment from short pieces of text, such as tweets, product reviews, or any user-generated content that may contain slang, emojis, and abbreviations. It uses a pre-built lexicon of words associated with sentiment values and applies a set of rules to calculate sentiment scores.''')
    st.markdown(''' - :blue-background[BERTopic] : BERTopic is a topic modeling technique that leverages the BERT transformer model to generate topics from a collection of documents. It uses the contextual embeddings of the BERT model to create document embeddings, which are then clustered to form topics. BERTopic is known for its ability to generate interpretable topics and handle large datasets efficiently.''')
    st.markdown(''' - :blue-background[Spacy NER] : spaCy is a popular NLP library that provides a Named Entity Recognition (NER) model to identify entities such as persons, organizations, locations, dates, and more in text data. The spaCy NER model is pre-trained on a large corpus of text data and can be used to extract valuable information from unstructured text.''')
    st.subheader("Other features:")
    st.markdown(''' - :blue-background[Word Cloud] : A word cloud is a visual representation of text data, where the size of each word indicates its frequency or importance in the text. Word clouds are often used to highlight key terms or themes in a document and provide a quick overview of the most common words in the text.''')
    st.markdown(''' - :blue-background[Graphical Representation] : The project provides graphical representations of the analyzed data, such as pie charts, bar graphs, and topic hierarchies, to help users visualize the results and identify patterns or trends in the data.''')
    st.markdown(''' - :blue-background[Download Data] : Users can download the analyzed data in CSV format for further analysis or reporting purposes. The download feature allows users to save the results of the analysis and share them with others easily.''')

st.sidebar.title("Upload File")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    try:    
        if uploaded_file.name.endswith('.csv'):
            data = organizer.csv_file(uploaded_file)
        elif uploaded_file.name.endswith('.pdf'):
            data = organizer.pdf_file(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            data = organizer.docx_file(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = organizer.excel_file(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type. Please upload a PDF, DOC, CSV or Excel file.")
        
        model = st.sidebar.radio("Select Model", ("VADAR", "BERTopic", "Spacy NER"))

        if st.sidebar.button("Show Analysis"):
            with st.spinner("Wait for it...", show_time=True):
                if model == 'VADAR':
                    st.title("Sentiment Analysis")

                    data['Text_Sentiment'] = data['text1'].apply(algorithms.sentiment_analyzer)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.header("Overall Sentiment Analysis")
                        fig, ax = plt.subplots()
                        ax.pie(data['Text_Sentiment'].value_counts(), labels = data['Text_Sentiment'].value_counts().index, autopct='%1.1f%%')
                        st.pyplot(fig)

                    with col2:
                        st.header("Word Cloud")
                        text = " ".join(data['text1'].astype(str))
                        wordcloud = WordCloud(width=800, height=800, colormap='binary').generate(text)
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    col1, col2, col3 = st.columns([2, 3, 2])
                    with col2:
                        st.header("Sentiment counts by Subreddit")
                        sentiment_counts = data
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
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col2:
                        st.header("Analyzed data")
                        st.write(topic_data)

                        csv = organizer.convert_df(topic_data)
                        st.download_button(
                            label="Download data",
                            data=csv,
                            file_name="data.csv",
                            mime="text/csv",
                        )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.header("Visual Representation")
                        st.write(topic_visual)

                    with col2:
                        st.header("Topic Hierarchy")
                        st.write(topic_hierarchy)

                elif model == "Spacy NER":
                    st.title("Spacy NER")

                    data = algorithms.spacy_ner(data)
                    
                    col1, col2, col3 = st.columns([2, 3, 2])
                    with col2:
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
                    fig, ax = plt.subplots()
                    ax.bar(entity_count.index, entity_count.values.flatten())
                    plt.xticks(rotation=90)
                    st.pyplot(fig)

    except Exception as e:
        st.sidebar.error(f"Error: {e}")

        
