import streamlit as st
import json

# Show title and description.
st.title("📄 Document question answering")

# Load data from data.json.
try:
    with open('data.json', 'r') as f:
        data = json.load(f)
    # Display the first 10 entries in the JSON file.
    st.write(data[:10])
except FileNotFoundError:
    st.error("The file 'data.json' was not found.")
except json.JSONDecodeError:
    st.error("Error decoding JSON from the file 'data.json'.")
