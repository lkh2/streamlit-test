import streamlit as st
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize
from streamlit_shadcn_ui import table

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
    common_style = "padding: 4px 8px; text-align: center; border-radius: 4px; display: inline-block; width: 100%;"
    styles = {
        'canceled': f"{common_style} background-color: #FFC5C5; color: #DF0404;",
        'failed': f"{common_style} background-color: #FFC5C5; color: #DF0404;",
        'suspended': f"{common_style} background-color: #FFC5C5; color: #DF0404;",
        'successful': f"{common_style} background-color: #16C09861; color: #00B087;",
        'live': f"{common_style} background-color: #E6F3FF; color: #0066CC;",
        'submitted': f"{common_style} background-color: #F0F0F0; color: #808080;"
    }
    return styles.get(val.lower(), '')

# Apply styling
def highlight_state(df):
    return pd.DataFrame('', index=df.index, columns=df.columns).style.apply(
        lambda x: [style_state(val) if col == 'State' else '' for col, val in x.items()], axis=1
    )

# Display the data with Shadcn UI table
st.title('Kickstarter Data Viewer')

# Convert DataFrame to list of dictionaries for Shadcn UI table
table_data = df.to_dict('records')

# Define table columns
columns = [
    {"key": "Project Name", "title": "Project Name"},
    {"key": "Creator", "title": "Creator"},
    {"key": "Pledged Amount", "title": "Pledged Amount"},
    {"key": "Country", "title": "Country"},
    {"key": "State", "title": "State"}
]

# Create custom cell renderer for State column
def render_state_cell(value):
    state_colors = {
        'canceled': {'bg': '#FFC5C5', 'text': '#DF0404'},
        'failed': {'bg': '#FFC5C5', 'text': '#DF0404'},
        'suspended': {'bg': '#FFC5C5', 'text': '#DF0404'},
        'successful': {'bg': '#16C09861', 'text': '#00B087'},
        'live': {'bg': '#E6F3FF', 'text': '#0066CC'},
        'submitted': {'bg': '#F0F0F0', 'text': '#808080'}
    }
    
    state = value.lower()
    colors = state_colors.get(state, {'bg': '#F0F0F0', 'text': '#808080'})
    
    return {
        "background": colors['bg'],
        "color": colors['text'],
        "padding": "4px 8px",
        "border-radius": "4px",
        "text-align": "center"
    }

# Display table with Shadcn UI
table(
    data=table_data,
    columns=columns,
    custom_column_styles={
        "State": render_state_cell
    },
    pagination=True,
    search=True,
    sorting=True,
    selection=True
)
