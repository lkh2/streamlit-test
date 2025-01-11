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
    items = collection.find()  # Removed the limit(200)
    
    # Convert MongoDB cursor to list and handle ObjectId
    items = [{**item, '_id': str(item['_id'])} for item in items]
    return items

items = get_data()

# Create DataFrame and restructure columns
df = json_normalize(items)
df = df[[
    'data.name',
    'data.creator.name',
    'data.converted_pledged_amount',
    'data.urls.web.project',
    'data.location.expanded_country',
    'data.state'
]].rename(columns={
    'data.name': 'Project Name',
    'data.creator.name': 'Creator',
    'data.converted_pledged_amount': 'Pledged Amount',
    'data.urls.web.project': 'Link',
    'data.location.expanded_country': 'Country',
    'data.state': 'State'
})

# Convert object columns to string
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].astype(str)

# Display the data
st.title('Kickstarter Data Viewer')
st.dataframe(df, use_container_width=True)
