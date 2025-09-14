import streamlit as st 
from structured_resume import structured_resume


st.title("Carrer guidence assistent")
st.header("I am here to help you about your carrer!")

from structured_resume import structured_resume

uploaded_file = st.file_uploader("Upload Resume", type="pdf")

if uploaded_file:
    result = structured_resume(uploaded_file)
    st.json(result)
