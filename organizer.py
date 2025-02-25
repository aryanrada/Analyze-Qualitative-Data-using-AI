import pandas as pd
import re
import base64
import streamlit as st

def fetch_data(subreddit,df):
    if subreddit != 'Overall':
        df = df[df['Subreddit'] == subreddit]
    return df

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def download_file(file_content, file_name, download_label="Download File"):
    """
    Creates a download link for a file in a Streamlit app.
    
    :param file_content: Content of the file as bytes
    :param file_name: Name of the file to be downloaded
    :param download_label: Label for the download button
    """
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file_name}">{download_label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def is_text_column(column_data):
    """
    Function to determine if a column contains paragraph or sentence-like content.
    This checks if the content is a string and contains sentence-like structures.
    """
    if column_data.apply(lambda x: isinstance(x, str)).all():
        if column_data.str.contains(r'[.!?]').any():
            return True
    return False

def filter_text_columns(df):
    """
    Filters out non-text columns, keeping only those that contain paragraph-like data.
    Also renames the columns to 'text', 'text1', 'text2', ...
    """
    valid_columns = []

    for column in df.columns:
        if is_text_column(df[column]):
            valid_columns.append(column)
    filtered_df = df[valid_columns]
    filtered_df.columns = [f'text{i+1}' for i in range(len(filtered_df.columns))]
    filtered_df = filtered_df.fillna("null")
    
    return filtered_df