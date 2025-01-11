import streamlit as st
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize
from bson import ObjectId

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

def convert_mongo_doc(doc):
    # Convert ObjectId to string
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc

# Pull data from the collection.
@st.cache_data(ttl=600)
def get_data():
    db = client[st.secrets["mongo"]["database"]]
    collection = db[st.secrets["mongo"]["collection"]]
    items = collection.find().limit(200)  # Limit to the first 200 entries
    items = [convert_mongo_doc(item) for item in items]  # Convert each document
    items = list(items)  # Make hashable for st.cache_data
    return items

items = get_data()

# Normalize the JSON data to flatten nested structures
df = json_normalize(items)

# Display the data in a table format with sortable columns
st.title('Kickstarter Data Viewer')
st.dataframe(df)
