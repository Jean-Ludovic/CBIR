import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"

st.title("CBIR Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file)

    if st.button("Search"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue())
        }

        r = requests.post(f"{API_URL}/gallery/search", files=files)

        st.json(r.json())
