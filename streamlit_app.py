import streamlit as st
import streamlit_shadcn_ui as ui
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
    items = collection.find().limit(500)
    
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

# Display the data with Shadcn UI table
st.title('Kickstarter Data Viewer')

# Define columns for the table
columns = [
    {"key": "Project Name", "title": "Project Name"},
    {"key": "Creator", "title": "Creator"},
    {"key": "Pledged Amount", "title": "Pledged Amount"},
    {"key": "Country", "title": "Country"},
    {"key": "State", "title": "State"}
]

# Add CSS for state styling
st.markdown("""
<style>
.state-cell {
    padding: 4px 8px;
    border-radius: 4px;
    text-align: center;
    display: inline-block;
}
.canceled, .failed, .suspended { 
    background-color: #FFC5C5;
    color: #DF0404;
}
.successful {
    background-color: #16C09861;
    color: #00B087;
}
.live {
    background-color: #E6F3FF;
    color: #0066CC;
}
.submitted {
    background-color: #F0F0F0;
    color: #808080;
}
</style>
""", unsafe_allow_html=True)

# Format state column with HTML
df['State'] = df['State'].apply(lambda x: f'<div class="state-cell {x.lower()}">{x}</div>')

# Display table
ui.table(
    data=df,
    columns=columns,
    maxHeight=500
)
