import re
import PyPDF2
from docx import Document
import pandas as pd
from nltk.tokenize import sent_tokenize
import streamlit as st
import unicodedata
import nltk
from nltk.corpus import stopwords

@st.cache_data
def convert_df(df):
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
    #url_pattern = re.compile(r'https?://\S+|www\.\S+')

    #def is_url_column(series):
    #    return series.astype(str).str.contains(url_pattern, na=False).any()
    
    #df = df.loc[:, ~df.apply(is_url_column)]
    return df

def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)

    return string

def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    
    return string

def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)
    
    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords

def clean(text):
    '''
    This function combines the above steps and added extra stop words to clean text
    '''
    return remove_stopwords(lemmatize(basic_clean(text)))