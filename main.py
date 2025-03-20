import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from io import BytesIO
import organizer
import algorithms

st.set_page_config(layout="wide")
#torch.hub.list('pytorch/vision', force_reload=True)
torch.classes.__path__ = []
#hi
st.sidebar.title("Analyze Qualitative Data using AI")

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
        #data = data.drop(columns=['Unnamed: 0', 'is_Original', 'Flair', 'URL'])
        #data["Title"].fillna("Null", inplace = True)
        #data["Body"].fillna("Null", inplace = True)
        #data["Comments"].fillna("Null", inplace = True)

        #subreddit = data['Subreddit'].unique().tolist()
        #subreddit.sort()
        #subreddit.insert(0,"Overall")
        #option = st.sidebar.selectbox("Select Subreddit", subreddit)
        #data = organizer.clean_dataset(data)
        model = st.sidebar.radio("Select Model", ("VADAR", "BERTopic", "Spacy NER"))

        if st.sidebar.button("Show Analysis"):
            #data = organizer.fetch_data(option,data)

            if model == 'VADAR':
                st.title("Sentiment Analysis")
                #data['Title_Sentiment'] = data['Title'].apply(algorithms.sentiment_analyzer)
                data['Text_Sentiment'] = data['text1'].apply(algorithms.sentiment_analyzer)
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Overall Sentiment Analysis")
                    fig, ax = plt.subplots()
                    ax.pie(data['Text_Sentiment'].value_counts(), labels = data['Text_Sentiment'].value_counts().index, autopct='%1.1f%%')
                    st.pyplot(fig)

                    #buf = BytesIO()
                    #fig.savefig(buf, format="png")
                    #buf.seek(0)

                    #with open("chart.png", "wb") as f:
                    #    f.write(buf.getvalue())

                    #with open("chart.png", "rb") as file:
                    #    st.download_button(
                    #        label="Download Chart",
                    #        data=fig,
                    #        file_name="chart.png",
                    #        mime="image/png",
                    #    )
                with col2:
                    st.header("Word Cloud")
                    text = " ".join(data['text1'].astype(str))
                    wordcloud = WordCloud(width=800, height=800, colormap='binary').generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                #st.header("Sentiment Analysis by Subreddit")
                #fig, ax = plt.subplots()
                #ax = sns.countplot(x='Subreddit', hue='Title_Sentiment', data=data)
                #ax = sns.countplot(hue='Title_Sentiment', data=data)
                #plt.xticks(rotation=90)
                #st.pyplot(fig)
                col1, col2, col3 = st.columns([2, 3, 2])
                with col2:
                    st.header("Sentiment counts by Subreddit")
                    #sentiment_counts = data['Text_Sentiment'].groupby(data['Subreddit']).value_counts().unstack().fillna(0)
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

                # fig, ax = plt.subplots()
                # topic_visual.write_image(ax)
                # buf = BytesIO()
                # fig.savefig(buf, format="png")
                # buf.seek(0)

                # with open("chart.png", "wb") as f:
                #     f.write(buf.getvalue())

                # with open("chart.png", "rb") as file:
                #     st.download_button(
                #         label="Download Chart",
                #         data=file,
                #         file_name="chart.png",
                #         mime="image/png",
                #     )
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
                #st.write(entity_count)
                fig, ax = plt.subplots()
                ax.bar(entity_count.index, entity_count.values.flatten())
                plt.xticks(rotation=90)
                st.pyplot(fig)

                #buf = BytesIO()
                #fig.savefig(buf, format="png")
                #buf.seek(0)

                #with open("chart.png", "wb") as f:
                #    f.write(buf.getvalue())

                #with open("chart.png", "rb") as file:
                #    st.download_button(
                #        label="Download Chart",
                #       data=file,
                #        file_name="chart.png",
                #        mime="image/png",
                #    )
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

        
