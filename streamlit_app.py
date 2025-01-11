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

# Function to style state with colored span
def style_state(state):
    state = state.lower()
    style_map = {
        'canceled': 'background: #FFC5C5; color: #DF0404; border-color: #DF0404',
        'failed': 'background: #FFC5C5; color: #DF0404; border-color: #DF0404',
        'suspended': 'background: #FFC5C5; color: #DF0404; border-color: #DF0404',
        'successful': 'background: #16C09861; color: #00B087; border-color: #00B087',
        'live': 'background: #E6F3FF; color: #0066CC; border-color: #0066CC',
        'submitted': 'background: #E6F3FF; color: #0066CC; border-color: #0066CC',
    }
    style = style_map.get(state, '')
    return f'<span style="width: 80%; text-align: center; padding: 4px 8px; border-radius: 4px; border: solid 1px; {style}">{state}</span>'

# Apply styling to State column
df['State'] = df['State'].apply(style_state)

# Convert DataFrame to HTML with styling
html_table = f"""
<style>
    table {{ border-collapse: collapse; width: 80%; }}
    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
</style>
{df.to_html(escape=False, index=False)}
"""

# Display the data
st.title('Kickstarter Data Viewer')
st.markdown(html_table, unsafe_allow_html=True)
