import pandas as pd
import streamlit as st

def fetch_data(subreddit,df):
    if subreddit != 'Overall':
        df = df[df['Subreddit'] == subreddit]
    return df