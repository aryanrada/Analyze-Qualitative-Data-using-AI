import streamlit as st

st.sidebar.title("Analyze Qualitative Data using AI")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    st.write(data)