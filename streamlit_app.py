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

# Create a custom component for state display
def create_state_element(state, key):
    state = state.lower()
    style_map = {
        'canceled': 'bg-red-100 text-red-800',
        'failed': 'bg-red-100 text-red-800',
        'suspended': 'bg-red-100 text-red-800',
        'successful': 'bg-green-100 text-green-800',
        'live': 'bg-blue-100 text-blue-800',
        'submitted': 'bg-blue-100 text-blue-800'
    }
    
    class_name = style_map.get(state, 'bg-gray-100 text-gray-800')
    class_name += ' px-2 py-1 rounded inline-block'
    
    return ui.element(
        "span",
        key=f"state_{key}",
        text=state,
        className=class_name
    )

# Display the data with Shadcn UI
st.title('Kickstarter Data Viewer')

with ui.card():
    # Display custom state elements for each row
    for idx, row in df.iterrows():
        with ui.accordion(key=f"row_{idx}", label=row['Project Name']):
            ui.text(f"Creator: {row['Creator']}")
            ui.text(f"Pledged Amount: {row['Pledged Amount']}")
            ui.text(f"Country: {row['Country']}")
            ui.text("State: ")
            create_state_element(row['State'], idx)

    # Display table with basic data
    ui.table(
        data=df,
        maxHeight=500
    )
