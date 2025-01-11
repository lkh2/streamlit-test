import streamlit as st
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize

# Initialize connection.
@st.cache_resource
def init_connection():
    mongo_connection_string = (
        f"mongodb+srv://{st.secrets['mongo']['username']}:"
        f"{st.secrets['mongo']['password']}@{st.secrets['mongo']['host']}/"
        f"{st.secrets['mongo']['database']}?retryWrites=true&w=majority"
    )
    return MongoClient(mongo_connection_string)

client = init_connection()

# Pull data from the collection.
@st.cache_data(ttl=600)
def get_data():
    db = client[st.secrets["mongo"]["database"]]
    collection = db[st.secrets["mongo"]["collection"]]
    items = collection.find().limit(200)
    # Convert ObjectId to string for each document
    processed_items = []
    for item in items:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
        processed_items.append(item)
    return processed_items

items = get_data()

# Normalize the JSON data to flatten nested structures
df = json_normalize(items)

# Display the data in a table format with sortable columns
st.title('Kickstarter Data Viewer')
st.dataframe(df)
