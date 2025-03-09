import base64
import re
import streamlit as st

def fetch_data(subreddit,df):
    if subreddit != 'Overall':
        df = df[df['Subreddit'] == subreddit]
    return df

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def remove_url_columns(data):
    """
    Removes any column containing URLs from a Pandas DataFrame.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame with URL columns removed.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    def is_url_column(series):
        return series.astype(str).str.contains(url_pattern, na=False).any()
    
    return data.loc[:, ~data.apply(is_url_column)]

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

def clean_dataset(df):
    """
    Removes URL columns and filters text columns from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: A cleaned DataFrame with URL columns removed and only text columns retained.
    """
    #df = remove_url_columns(df)
    df = filter_text_columns(df)
    
    return df

    #url_pattern = re.compile(r'https?://\S+|www\.\S+')

    #def is_url_column(series):
    #    return series.astype(str).str.contains(url_pattern, na=False).any()
    
    #df = df.loc[:, ~df.apply(is_url_column)]