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

# Create styling function
def style_state(val):
    common_style = "padding: 4px 8px; text-align: center; border-radius: 1px; display: inline-block; width: 80%;"
    styles = {
        'canceled': f"background-color: #FFC5C5; color: #DF0404; border: 1px solid #DF0404;",
        'failed': f"background-color: #FFC5C5; color: #DF0404; border: 1px solid #DF0404;",
        'suspended': f"background-color: #FFC5C5; color: #DF0404; border: 1px solid #DF0404;",
        'successful': f"background-color: #16C09861; color: #00B087; border: 1px solid #00B087;",
        'live': f"background-color: #E6F3FF; color: #0066CC; border: 1px solid #0066CC;",
        'submitted': f"background-color: #F0F0F0; color: #808080; border: 1px solid #808080;"
    }
    style = styles.get(val.lower(), '')
    return f'<span style="{common_style} {style}">{val}</span>'

# Apply styling
def highlight_state(df):
    return pd.DataFrame('', index=df.index, columns=df.columns).style.format(
        {'State': lambda x: style_state(x)}
    )

# Display the data with styling
st.title('Kickstarter Data Viewer')
styled_df = df.style.format({'State': lambda x: style_state(x)}).hide_index()
st.dataframe(styled_df, use_container_width=True, unsafe_allow_html=True)
