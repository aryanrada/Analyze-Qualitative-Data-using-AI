import streamlit as st
import pandas as pd

st.sidebar.title("Analyze Qualitative Data using AI")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    #bytes_data = uploaded_file.getvalue()
    #data = bytes_data.decode("utf-8")
    #data = pd.DataFrame(data)
    data = data.drop(columns=['Unnamed: 0', 'is_Original', 'Flair', 'URL'])
    st.write(data)