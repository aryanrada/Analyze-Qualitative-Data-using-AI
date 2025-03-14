import re
import PyPDF2
from docx import Document
import pandas as pd
from nltk.tokenize import sent_tokenize
import streamlit as st

#def fetch_data(subreddit,df):
#    if subreddit != 'Overall':
#        df = df[df['Subreddit'] == subreddit]
#    return df

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def csv_file(file):
    data = pd.read_csv(file)
    data = clean_dataset(data)
    return data

def excel_file(file):
    data = pd.read_excel(file)
    data = clean_dataset(data)
    return data

def pdf_file(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'

    sentences = sent_tokenize(text)
    df = pd.DataFrame({'Sentences': sentences})
    data = clean_dataset(df)
    return data

def docx_file(file):
    doc = Document(file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)

    sentences = sent_tokenize('\n'.join(text))
    df = pd.DataFrame({'Sentences': sentences})
    data = clean_dataset(df)
    return data

def remove_url_columns(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    def is_url_column(series):
        return series.astype(str).str.contains(url_pattern, na=False).any()
    
    return data.loc[:, ~data.apply(is_url_column)]

def is_text_column(column_data):
    #Function to determine if a column contains paragraph or sentence-like content. This checks if the content is a string and contains sentence-like structures.
    if column_data.apply(lambda x: isinstance(x, str)).all():
        if column_data.str.contains(r'[.!?]').any():
            return True
    return False

def filter_text_columns(df):
    #Filters out non-text columns, keeping only those that contain paragraph-like data. Also renames the columns to 'text', 'text1', 'text2', ...
    valid_columns = []

    for column in df.columns:
        if is_text_column(df[column]):
            valid_columns.append(column)
    filtered_df = df[valid_columns]
    filtered_df.columns = [f'text{i+1}' for i in range(len(filtered_df.columns))]
    filtered_df = filtered_df.fillna("null")
    
    return filtered_df

def clean_dataset(df):
    #df = remove_url_columns(df)
    df = filter_text_columns(df)
    
    return df

    #url_pattern = re.compile(r'https?://\S+|www\.\S+')

    #def is_url_column(series):
    #    return series.astype(str).str.contains(url_pattern, na=False).any()
    
    #df = df.loc[:, ~df.apply(is_url_column)]