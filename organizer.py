import pandas as pd

def fetch_data(subreddit,df):
    if subreddit != 'Overall':
        df = df[df['Subreddit'] == subreddit]
    return df
