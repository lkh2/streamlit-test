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
        'canceled': {'bg': '#FFC5C5', 'color': '#DF0404'},
        'failed': {'bg': '#FFC5C5', 'color': '#DF0404'},
        'suspended': {'bg': '#FFC5C5', 'color': '#DF0404'},
        'successful': {'bg': '#16C09861', 'color': '#00B087'},
        'live': {'bg': '#E6F3FF', 'color': '#0066CC'},
        'submitted': {'bg': '#E6F3FF', 'color': '#0066CC'}
    }
    colors = style_map.get(state, {'bg': '#F0F0F0', 'color': '#808080'})
    return f'''<span style="
        display: inline-block;
        width: 100px;
        padding: 4px 8px;
        border-radius: 4px;
        background: {colors['bg']};
        color: {colors['color']};
        border: 1px solid {colors['color']};
        text-align: center;
        ">{state}</span>'''

# Apply styling to State column
df['State'] = df['State'].apply(style_state)

# Convert DataFrame to HTML with styling
html_table = f"""
<style>
    .table-container {{
        display: flex;
        justify-content: center;
        padding: 20px;
    }}
    table {{
        border-collapse: collapse;
        width: 80%;
        max-width: 1200px;
    }}
    th, td {{
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    th {{
        background-color: #f8f9fa;
        font-weight: 600;
    }}
    td {{
        vertical-align: middle;
    }}
</style>
<div class="table-container">
    {df.to_html(escape=False, index=False)}
</div>
"""

# Display the data
st.title('Kickstarter Data Viewer')
st.markdown(html_table, unsafe_allow_html=True)
