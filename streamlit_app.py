import time
import streamlit as st
import streamlit.components.v1 as components
import tempfile, os
import pandas as pd
from pandas import json_normalize
from streamlit_js_eval import get_geolocation
import json
import numpy as np
import gzip
import glob
import polars as pl

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #2A5D4E 0%, #65897F 50%, #2A5D4E 100%);
        }  
        [data-testid="stHeader"] {
            background: transparent;
        }  
    </style>
    """,
    unsafe_allow_html=True
)

def generate_component(name, template="", script=""):
    def html():
        return f"""
            <!DOCTYPE html>
            <html lang="en">
                <head>
                    <link href='https://fonts.googleapis.com/css?family=Poppins' rel='stylesheet'>
                    <link href='https://fonts.googleapis.com/css?family=Playfair Display' rel='stylesheet'>
                    <meta charset="UTF-8" />
                    <title>{name}</title>
                    <script>
                        function sendMessageToStreamlitClient(type, data) {{
                            const outData = Object.assign({{
                                isStreamlitMessage: true,
                                type: type,
                            }}, data);
                            window.parent.postMessage(outData, "*");
                        }}

                        const Streamlit = {{
                            setComponentReady: function() {{
                                sendMessageToStreamlitClient("streamlit:componentReady", {{apiVersion: 1}});
                            }},
                            setFrameHeight: function(height) {{
                                sendMessageToStreamlitClient("streamlit:setFrameHeight", {{height: height}});
                            }},
                            setComponentValue: function(value) {{
                                sendMessageToStreamlitClient("streamlit:setComponentValue", {{value: value}});
                            }},
                            RENDER_EVENT: "streamlit:render",
                            events: {{
                                addEventListener: function(type, callback) {{
                                    window.addEventListener("message", function(event) {{
                                        if (event.data.type === type) {{
                                            event.detail = event.data
                                            callback(event);
                                        }}
                                    }});
                                }}
                            }}
                        }}
                    </script>
                </head>
            <body>
            {template}
            </body>
            <script>
                {script}
            </script>
            </html>
        """

    dir = f"{tempfile.gettempdir()}/{name}"
    if not os.path.isdir(dir): os.mkdir(dir)
    fname = f'{dir}/index.html'
    with open(fname, 'w') as f:
        f.write(html())
    
    func = components.declare_component(name, path=str(dir))
    def f(**params):
        component_value = func(**params)
        return component_value
    return f

# Replace MongoDB connection with Polars-powered Parquet processing
@st.cache_data(ttl=600)
def load_data_from_parquet_chunks():
    """
    Load data from compressed parquet chunks by first combining them into a complete file
    """
    # Find all parquet chunk files
    chunk_files = glob.glob("parquet_gz_chunks/*.part")
    
    if not chunk_files:
        st.error("No parquet chunks found in parquet_gz_chunks folder")
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Found {len(chunk_files)} chunk files. Combining chunks...")
    
    # Create a temporary file to store the combined chunks
    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet.gz') as combined_file:
        combined_filename = combined_file.name
        
        # Sort the chunks to ensure they're processed in the correct order
        chunk_files = sorted(chunk_files)
        
        # First combine all chunks into a single file
        for i, chunk_file in enumerate(chunk_files):
            try:
                with open(chunk_file, 'rb') as f:
                    combined_file.write(f.read())
                progress_bar.progress((i + 1) / (2 * len(chunk_files)))  # First half of progress is combining
                status_text.text(f"Combined chunk {i+1}/{len(chunk_files)}")
            except Exception as e:
                st.warning(f"Error reading chunk {chunk_file}: {str(e)}")
    
    # Create another temporary file for the decompressed parquet
    decompressed_filename = None
    
    try:
        status_text.text("Decompressing combined file...")
        
        # Create a temporary file for the decompressed parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as decompressed_file:
            decompressed_filename = decompressed_file.name
            
            # Decompress the combined gzip file
            with gzip.open(combined_filename, 'rb') as gz_file:
                decompressed_file.write(gz_file.read())
        
        # Read the decompressed parquet file
        status_text.text("Reading parquet data...")
        progress_bar.progress(0.75)  # 75% progress after decompression
        
        # Read the parquet file
        df = pd.read_parquet(decompressed_filename, engine='pyarrow')
        
        # Check column format and handle both prefixed and non-prefixed columns
        column_format = 'data_prefix' if 'data.state' in df.columns else 'no_prefix'
        status_text.text(f"Detected column format: {column_format}")
        
        if column_format == 'no_prefix':
            # Add 'data.' prefix to all columns except those that already have structures
            rename_map = {}
            for col in df.columns:
                if '.' not in col and col != 'run_id':
                    rename_map[col] = f'data.{col}'
            
            # Only rename if there are columns to rename
            if rename_map:
                df = df.rename(columns=rename_map)
                status_text.text("Added 'data.' prefix to column names")
        
        # Filter for successful projects if not already filtered
        if 'data.state' in df.columns and 'successful' in df['data.state'].unique():
            original_len = len(df)
            df = df[df['data.state'] == 'successful']
            if len(df) < original_len:
                status_text.text(f"Filtered to {len(df)} successful projects")
        
        # Limit the number of rows if needed
        limit = 999999
        if len(df) > limit:
            df = df.iloc[:limit]
            status_text.text(f"Limited to {limit} rows")
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"Successfully loaded {len(df)} items")
        
        # Convert to the expected format (list of dictionaries)
        items = df.to_dict(orient='records')
        return items
        
    except Exception as e:
        st.error(f"Error processing combined parquet file: {str(e)}")
        return []
    finally:
        # Clean up temporary files
        try:
            if combined_filename:
                os.unlink(combined_filename)
            if decompressed_filename:
                os.unlink(decompressed_filename)
        except Exception as e:
            st.warning(f"Error cleaning up temporary files: {str(e)}")

# Function to create dummy data if everything else fails
def create_dummy_data():
    """Create some dummy kickstarter data for testing"""
    st.warning("Loading dummy data for testing")
    dummy_items = []
    
    for i in range(100):
        # Create a random date within last 5 years
        created_at = int((pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 1825))).timestamp())
        deadline = created_at + np.random.randint(15, 60) * 86400  # 15-60 days later
        
        goal = np.random.randint(1000, 50000)
        pledged = np.random.randint(0, int(goal * 2))
        backers = np.random.randint(0, 500)
        
        dummy_item = {
            "data": {
                "name": f"Test Project {i}",
                "creator": {"name": f"Creator {i % 10}"},
                "goal": float(goal),
                "usd_exchange_rate": 1.0,
                "converted_pledged_amount": float(pledged),
                "backers_count": backers,
                "created_at": created_at,
                "deadline": deadline,
                "state": "successful",
                "urls": {"web": {"project": f"https://example.com/project{i}"}},
                "location": {
                    "country": "US",
                    "expanded_country": "United States"
                },
                "category": {
                    "parent_name": ["Technology", "Art", "Games", "Design", "Food"][i % 5],
                    "name": f"Subcategory {i % 10}"
                },
                "staff_pick": bool(i % 5 == 0)
            }
        }
        dummy_items.append(dummy_item)
    
    return dummy_items

# Load data using the memory-efficient Polars function
items = load_data_from_parquet_chunks()

# Create DataFrame and inspect the structure of items
if len(items) > 0:
    pass

# Create DataFrame with json_normalize and check keys
try:
    df = json_normalize(items)
except Exception as e:
    st.error(f"Error in json_normalize: {str(e)}")
    # Fallback: create dataframe directly 
    df = pd.DataFrame(items)

# If we have nested 'data' dictionaries, handle them
if 'data' in df.columns and len(df) > 0:
    if isinstance(df['data'].iloc[0], dict):
        # Extract nested dictionaries
        data_df = pd.json_normalize(df['data'].tolist())
        # Combine with original df
        for col in data_df.columns:
            df[f'data.{col}'] = data_df[col]
        # Drop the original nested column
        df = df.drop(columns=['data'])

# Now proceed with column normalization...

# Inspect available columns and print for debugging
# st.write("Available columns:", df.columns.tolist())

# Define a helper function to safely access columns that might have different naming
def safe_column_access(df, possible_names):
    """Try multiple possible column names and return the first one that exists"""
    for col in possible_names:
        if col in df.columns:
            return col
    # If no matching column found, return None
    st.warning(f"Could not find any column matching: {possible_names}")
    return None

# Find the correct column names
goal_col = safe_column_access(df, ['data.goal', 'data_goal', 'goal'])
exchange_rate_col = safe_column_access(df, ['data.usd_exchange_rate', 'data_usd_exchange_rate', 'usd_exchange_rate'])
pledged_col = safe_column_access(df, ['data.converted_pledged_amount', 'data_converted_pledged_amount', 'converted_pledged_amount'])
created_col = safe_column_access(df, ['data.created_at', 'data_created_at', 'created_at'])
deadline_col = safe_column_access(df, ['data.deadline', 'data_deadline', 'deadline'])
backers_col = safe_column_access(df, ['data.backers_count', 'data_backers_count', 'backers_count'])

# Calculate and store raw values - only if columns exist
if goal_col and exchange_rate_col:
    df['Raw Goal'] = df[goal_col].fillna(0).astype(float) * df[exchange_rate_col].fillna(1.0).astype(float)
    df['Raw Goal'] = df['Raw Goal'].apply(lambda x: max(1.0, x))
else:
    df['Raw Goal'] = 1.0  # Default value if columns not found

if pledged_col:
    df['Raw Pledged'] = df[pledged_col].fillna(0).astype(float)
else:
    df['Raw Pledged'] = 0.0  # Default value if column not found

# Calculate Raw Raised with special handling for zero pledged amount
df['Raw Raised'] = df.apply(
    lambda row: 0.0 if row['Raw Pledged'] == 0 or row['Raw Goal'] == 0
    else (row['Raw Pledged'] / row['Raw Goal']) * 100, 
    axis=1
)

if created_col:
    df['Raw Date'] = pd.to_datetime(df[created_col], unit='s')
else:
    df['Raw Date'] = pd.to_datetime('2020-01-01')  # Default fallback date

# Convert deadline to datetime and format display columns
if deadline_col:
    df['Raw Deadline'] = pd.to_datetime(df[deadline_col], unit='s')
    df['Deadline'] = df['Raw Deadline'].dt.strftime('%Y-%m-%d')
else:
    df['Raw Deadline'] = pd.to_datetime('2020-12-31')  # Default fallback date
    df['Deadline'] = '2020-12-31'

# Backer count with null handling
if backers_col:
    df['Backer Count'] = df[backers_col].fillna(0).astype(int)
else:
    df['Backer Count'] = 0  # Default value if column not found

# Format display columns - Add null handling
df['Goal'] = df['Raw Goal'].fillna(0).round(2).map(lambda x: f"${x:,.2f}")
df['Pledged Amount'] = df['Raw Pledged'].fillna(0).map(lambda x: f"${int(x):,}")
df['%Raised'] = df['Raw Raised'].fillna(0).map(lambda x: f"{x:.1f}%")
df['Date'] = df['Raw Date'].dt.strftime('%Y-%m-%d')

# Continue working with the remaining columns similarly
# Find other required columns
name_col = safe_column_access(df, ['data.name', 'data_name', 'name'])
creator_col = safe_column_access(df, ['data.creator.name', 'data_creator_name', 'creator_name'])
link_col = safe_column_access(df, ['data.urls.web.project', 'data_urls_web_project', 'urls_web_project'])
country_expanded_col = safe_column_access(df, ['data.location.expanded_country', 'data_location_expanded_country', 'location_expanded_country'])
state_col = safe_column_access(df, ['data.state', 'data_state', 'state'])
category_col = safe_column_access(df, ['data.category.parent_name', 'data_category_parent_name', 'category_parent_name'])
subcategory_col = safe_column_access(df, ['data.category.name', 'data_category_name', 'category_name'])
country_code_col = safe_column_access(df, ['data.location.country', 'data_location_country', 'location_country'])
staff_pick_col = safe_column_access(df, ['data.staff_pick', 'data_staff_pick', 'staff_pick'])

# Create a new DataFrame with only the columns we need
new_columns = {}

# Add columns with fallbacks
new_columns['Project Name'] = df[name_col] if name_col else 'Unknown Project'
new_columns['Creator'] = df[creator_col] if creator_col else 'Unknown Creator'
new_columns['Pledged Amount'] = df['Pledged Amount']
new_columns['Link'] = df[link_col] if link_col else '#'
new_columns['Country'] = df[country_expanded_col] if country_expanded_col else 'Unknown'
new_columns['State'] = df[state_col] if state_col else 'unknown'
new_columns['Category'] = df[category_col] if category_col else 'Other'
new_columns['Subcategory'] = df[subcategory_col] if subcategory_col else 'Other'
new_columns['Date'] = df['Date']
new_columns['Deadline'] = df['Deadline']
new_columns['Goal'] = df['Goal']
new_columns['%Raised'] = df['%Raised']
new_columns['Raw Goal'] = df['Raw Goal']
new_columns['Raw Pledged'] = df['Raw Pledged']
new_columns['Raw Raised'] = df['Raw Raised']
new_columns['Raw Date'] = df['Raw Date']
new_columns['Raw Deadline'] = df['Raw Deadline']
new_columns['Backer Count'] = df['Backer Count']
new_columns['Country Code'] = df[country_code_col] if country_code_col else 'US'
new_columns['Staff Pick'] = df[staff_pick_col] if staff_pick_col else False

# Create new dataframe with the correct columns
df = pd.DataFrame(new_columns)

# Convert remaining object columns to string
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].astype(str)

# Function to style state with colored span
def style_state(state):
    state = state.lower()
    return f'<div class="state_cell state-{state}">{state}</div>'

# Apply styling to State column
df['State'] = df['State'].apply(style_state)

# After creating the initial DataFrame, add country coordinates
@st.cache_data
def load_country_data():
    # Downloaded from https://developers.google.com/public-data/docs/canonical/countries_csv
    country_df = pd.read_csv('country.csv')
    return country_df

# Add latitude/longitude from country data
country_data = load_country_data()
df = df.merge(country_data[['country', 'latitude', 'longitude']], 
              left_on='Country Code', 
              right_on='country', 
              how='left')

# Add geolocation call before data processing
loc = get_geolocation()
user_location = None

if (loc and 'coords' in loc):
    with st.spinner('Updating table with your location...'):
        user_location = {
            'latitude': loc['coords']['latitude'], 
            'longitude': loc['coords']['longitude']
        }
        time.sleep(1)
    loading_success = st.success("Location received successfully!")
    time.sleep(1.5)
    loading_success.empty()

# Add function to calculate distances
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1  # Fixed: Calculate dlon correctly
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# After merging with country data, calculate distances if location is available
if user_location:
    df['Distance'] = df.apply(
        lambda row: calculate_distance(
            user_location['latitude'],
            user_location['longitude'],
            float(row['latitude']) if pd.notna(row['latitude']) else 0,  # Handle NaN values
            float(row['longitude']) if pd.notna(row['longitude']) else 0  # Handle NaN values
        ) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else float('inf'),
        axis=1
    ).astype(float)  # Ensure Distance is float type
else:
    df['Distance'] = float('inf')  # Set as float infinity

# Sort DataFrame by Distance initially to verify values
df = df.sort_values('Distance')

# Calculate popularity score components
now = pd.Timestamp.now()
max_days = (now - df['Raw Date']).dt.total_seconds().max() / (24*60*60)
time_factor = 1 - ((now - df['Raw Date']).dt.total_seconds() / (24*60*60) / max_days)

# Cap percentage raised at 500% to prevent extreme outliers
capped_percentage = df['Raw Raised'].clip(upper=500)

# Normalize components to 0-1 scale
normalized_backers = (df['Backer Count'] - df['Backer Count'].min()) / (df['Backer Count'].max() - df['Backer Count'].min())
normalized_pledged = (df['Raw Pledged'] - df['Raw Pledged'].min()) / (df['Raw Pledged'].max() - df['Raw Pledged'].min())  # Fixed this line
normalized_percentage = (capped_percentage - capped_percentage.min()) / (capped_percentage.max() - capped_percentage.min())

# Calculate popularity score
df['Popularity Score'] = (
    normalized_backers * 0.25 +      # Backer count (25% weight, reduced from 35%)
    normalized_pledged * 0.35 +      # Pledged amount (35% weight, increased from 25%)
    normalized_percentage * 0.20 +    # Percentage raised (20% weight, unchanged)
    time_factor * 0.10 +             # Time factor (10% weight, unchanged)
    df['Staff Pick'].astype(int) * 0.10  # Staff pick (10% weight, unchanged)
)

def generate_table_html(df):
    # Define visible and hidden columns
    visible_columns = ['Project Name', 'Creator', 'Pledged Amount', 'Link', 'Country', 'State']
    
    # Generate header for visible columns only
    header_html = ''.join(f'<th scope="col">{column}</th>' for column in visible_columns)
    
    # Generate table rows with raw values in data attributes
    rows_html = ''
    for _, row in df.iterrows():
        # Add data attributes to each row for filtering
        data_attrs = f'''
            data-category="{row['Category']}"
            data-subcategory="{row['Subcategory']}"
            data-pledged="{row['Raw Pledged']}"
            data-goal="{row['Raw Goal']}"
            data-raised="{row['Raw Raised']}"
            data-date="{row['Raw Date'].strftime('%Y-%m-%d')}"
            data-deadline="{row['Raw Deadline'].strftime('%Y-%m-%d')}"
            data-backers="{row['Backer Count']}"
            data-latitude="{row['latitude']}"
            data-longitude="{row['longitude']}"
            data-country-code="{row['Country Code']}"
            data-distance="{row['Distance']:.2f}"
            data-staff-pick="{str(row['Staff Pick']).lower()}"
            data-popularity="{row['Popularity Score']:.6f}"
        '''
        
        # Create visible cells with special handling for Link column
        visible_cells = ''
        for col in visible_columns:
            if (col == 'Link'):
                url = row[col]
                visible_cells += f'<td><a href="{url}" target="_blank">{url}</a></td>'
            else:
                visible_cells += f'<td>{row[col]}</td>'
        
        # Add the row with data attributes
        rows_html += f'<tr class="table-row" {data_attrs}>{visible_cells}</tr>'
    
    return header_html, rows_html

# Generate table HTML
header_html, rows_html = generate_table_html(df)

# Calculate min/max values from the DataFrame
min_pledged = int(df['Raw Pledged'].min())
max_pledged = int(df['Raw Pledged'].max())
min_goal = int(df['Raw Goal'].min())
max_goal = int(df['Raw Goal'].max())
min_raised = int(df['Raw Raised'].min())
max_raised = int(df['Raw Raised'].max())

# After loading data and before generating table, prepare filter options
def get_filter_options(df):
    # Make sure 'All Subcategories' is first, then sort the rest
    subcategories = df['Subcategory'].unique().tolist()
    sorted_subcategories = sorted(subcategories)
    
    # Extract states without HTML formatting
    states = df['State'].str.extract(r'state-(\w+)')[0].unique().tolist()
    states = [state.title() for state in states]  # Capitalize first letter
    
    return {
        'categories': sorted(['All Categories'] + df['Category'].unique().tolist()),
        'subcategories': ['All Subcategories'] + sorted_subcategories,
        'countries': sorted(['All Countries'] + df['Country'].unique().tolist()),
        'states': sorted(['All States'] + states),
        'date_ranges': [
            'All Time',
            'Last Month',
            'Last 6 Months',
            'Last Year',
            'Last 5 Years',
            'Last 10 Years'
        ]
    }

filter_options = get_filter_options(df)

# Update template to include filter controls with default subcategory
template = f"""
<script>
    // Make user location available to JavaScript
    window.userLocation = {json.dumps(user_location) if user_location else 'null'};
    window.hasLocation = {json.dumps(bool(user_location))};
</script>
<div class="title-wrapper">
    <span>Explore Successful Projects</span>
</div>
<div class="filter-wrapper">
    <div class="reset-wrapper">
        <button class="reset-button" id="resetFilters">
            <span>Default</span>
        </button>
    </div>
    <div class="filter-controls">
        <div class="filter-row">
            <span class="filter-label">Explore</span>
            <div class="multi-select-dropdown">
                <button id="categoryFilterBtn" class="filter-select multi-select-btn">Categories</button>
                <div class="multi-select-content">
                    {' '.join(f'<div class="category-option" data-value="{opt}">{opt}</div>' for opt in filter_options['categories'])}
                </div>
            </div>
            <span class="filter-label">&</span>
            <div class="multi-select-dropdown">
                <button id="subcategoryFilterBtn" class="filter-select multi-select-btn">Subcategories</button>
                <div class="multi-select-content">
                    {' '.join(f'<div class="subcategory-option" data-value="{opt}">{opt}</div>' for opt in filter_options['subcategories'])}
                </div>
            </div>
            <span class="filter-label">Projects On</span>
            <div class="multi-select-dropdown">
                <button id="countryFilterBtn" class="filter-select multi-select-btn">Countries</button>
                <div class="multi-select-content">
                    {' '.join(f'<div class="country-option" data-value="{opt}">{opt}</div>' for opt in filter_options['countries'])}
                </div>
            </div>
            <span class="filter-label">Sorted By</span>
            <select id="sortFilter" class="filter-select">
                <option value="popularity" selected>Most Popular</option>
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="mostfunded">Most Funded</option>
                <option value="mostbacked">Most Backed</option>
                <option value="enddate">End Date</option>
                <option value="nearme">Near Me</option>
            </select>
        </div>
        <div class="filter-row">
            <span class="filter-label">More Flexible, Dynamic Search:</span>
            <div class="multi-select-dropdown">
                <button id="stateFilterBtn" class="filter-select multi-select-btn">States</button>
                <div class="multi-select-content">
                    {' '.join(f'<div class="state-option" data-value="{opt}">{opt}</div>' for opt in filter_options['states'])}
                </div>
            </div>
            <div class="range-dropdown">
                <button class="filter-select">Pledged Amount Range</button>
                <div class="range-content">
                    <div class="range-container">
                        <div class="sliders-control">
                            <input id="fromSlider" type="range" value="{min_pledged}" min="{min_pledged}" max="{max_pledged}"/>
                            <input id="toSlider" type="range" value="{max_pledged}" min="{min_pledged}" max="{max_pledged}"/>
                        </div>
                        <div class="form-control">
                            <div class="form-control-container">
                                <span class="form-control-label">Min $</span>
                                <input class="form-control-input" type="number" id="fromInput" value="{min_pledged}" min="{min_pledged}" max="{max_pledged}"/>
                            </div>
                            <div class="form-control-container">
                                <span class="form-control-label">Max $</span>
                                <input class="form-control-input" type="number" id="toInput" value="{max_pledged}" min="{min_pledged}" max="{max_pledged}"/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="range-dropdown">
                <button class="filter-select">Goal Amount Range</button>
                <div class="range-content">
                    <div class="range-container">
                        <div class="sliders-control">
                            <input id="goalFromSlider" type="range" value="{min_goal}" min="{min_goal}" max="{max_goal + 1}"/>
                            <input id="goalToSlider" type="range" value="{max_goal + 1}" min="{min_goal}" max="{max_goal + 1}"/>
                        </div>
                        <div class="form-control">
                            <div class="form-control-container">
                                <span class="form-control-label">Min $</span>
                                <input class="form-control-input" type="number" id="goalFromInput" value="{min_goal}" min="{min_goal}" max="{max_goal + 1}"/>
                            </div>
                            <div class="form-control-container">
                                <span class="form-control-label">Max $</span>
                                <input class="form-control-input" type="number" id="goalToInput" value="{max_goal + 1}" min="{min_goal}" max="{max_goal + 1}"/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="range-dropdown">
                <button class="filter-select">Percentage Raised Range</button>
                <div class="range-content">
                    <div class="range-container">
                        <div class="sliders-control">
                            <input id="raisedFromSlider" type="range" value="{min_raised}" min="{min_raised}" max="{max_raised}"/>
                            <input id="raisedToSlider" type="range" value="{max_raised}" min="{min_raised}" max="{max_raised}"/>
                        </div>
                        <div class="form-control">
                            <div class="form-control-container">
                                <span class="form-control-label">Min %</span>
                                <input class="form-control-input" type="number" id="raisedFromInput" value="{min_raised}" min="{min_raised}" max="{max_raised}"/>
                            </div>
                            <div class="form-control-container">
                                <span class="form-control-label">Max %</span>
                                <input class="form-control-input" type="number" id="raisedToInput" value="{max_raised}" min="{min_raised}" max="{max_raised}"/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <select id="dateFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['date_ranges'])}
            </select>
        </div>
    </div>
</div>
<div class="table-wrapper">
    <div class="table-controls">
        <span class="filtered-text">Filtered Projects</span>
        <input type="text" id="table-search" class="search-input" placeholder="Search table...">
    </div>
    <div class="table-container">
        <table id="data-table">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    <div class="pagination-controls">
        <button id="prev-page" class="page-btn" aria-label="Previous page">&lt;</button>
        <div id="page-numbers" class="page-numbers"></div>
        <button id="next-page" class="page-btn" aria-label="Next page">&gt;</button>
    </div>
</div>
<script>
    // Make user location available to JavaScript
    window.userLocation = {user_location if user_location else 'null'};
</script>
"""

# Add new CSS styles
css = """
<style> 
    .title-wrapper {
        width: 100%;
        text-align: center;    
        margin-bottom: 25px;
    }
    
    .title-wrapper span {
        color: white;
        font-family: 'Playfair Display';
        font-weight: 500;
        font-size: 70px;
    }
    
    .table-controls { 
        position: sticky; 
        top: 0; 
        background: #ffffff; 
        z-index: 2; 
        padding: 0 20px; 
        border-bottom: 1px solid #eee; 
        height: 60px; 
        display: flex; 
        align-items: center; 
        justify-content: space-between;
        margin-bottom: 1rem; 
        border-radius: 20px; 
    }
    
    .table-container { 
        position: relative;
        flex: 1;
        padding: 20px; 
        background: #ffffff; 
        overflow-y: auto;
        transition: height 0.3s ease;
        z-index: 3;
    }
    
    table { 
        border-collapse: collapse; 
        width: 100%; 
        background: #ffffff; 
        table-layout: fixed; 
    }

    /* Column width specifications */
    th[scope="col"]:nth-child(1) { width: 25%; }  /* Project Name - 2 parts */
    th[scope="col"]:nth-child(2) { width: 12.5%; }  /* Creator - 1 part */
    th[scope="col"]:nth-child(3) { width: 120px; }  /* Pledged Amount - fixed */
    th[scope="col"]:nth-child(4) { width: 25%; }  /* Link - 2 parts */
    th[scope="col"]:nth-child(5) { width: 12.5%; }  /* Country - 1 part */
    th[scope="col"]:nth-child(6) { width: 120px; }  /* State - fixed */

    th { 
        background: #ffffff; 
        position: sticky; 
        top: 0; 
        z-index: 1; 
        padding: 12px 8px; 
        font-weight: 500; 
        font-family: 'Poppins'; 
        font-size: 14px; 
        color: #B5B7C0; 
        text-align: left; 
    }
    
    th:last-child { 
        text-align: center; 
    }

    td { 
        padding: 8px; 
        text-align: left; 
        border-bottom: 1px solid #ddd; 
        white-space: nowrap;
        font-family: 'Poppins';
        font-size: 14px;
        overflow-x: auto;
        -ms-overflow-style: none;
        overflow: -moz-scrollbars-none;
        scrollbar-width: none;
    }
    
    td::-webkit-scrollbar {
        display: none;  /* Safari and Chrome */
    }

    td:last-child { 
        width: 120px; 
        max-width: 120px; 
        text-align: center; 
    }

    .state_cell { 
        width: 100px; 
        max-width: 100px; 
        margin: 0 auto; 
        padding: 3px 5px; 
        text-align: center; 
        border-radius: 4px; 
        border: solid 1px; 
        display: inline-block; 
    }

    .state-canceled, .state-failed, .state-suspended { 
        background: #FFC5C5; 
        color: #DF0404; 
        border-color: #DF0404; 
    }
    
    .state-successful { 
        background: #16C09861; 
        color: #00B087; 
        border-color: #00B087; 
    }
    
    .state-live, .state-submitted, .state-started { 
        background: #E6F3FF; 
        color: #0066CC; 
        border-color: #0066CC; 
    }

    .table-wrapper { 
        position: relative;
        display: flex;
        flex-direction: column;
        max-width: 100%; 
        background: linear-gradient(180deg, #ffffff 15%, transparent 100%); 
        border-radius: 20px; 
        overflow: visible;
        transition: height 0.3s ease;
    }

    .search-input { 
        padding: 8px 12px; 
        border: 1px solid #ddd; 
        border-radius: 20px; 
        width: 200px; 
        font-size: 10px; 
        font-family: 'Poppins'; 
    }

    .search-input:focus { 
        outline: none; 
        border-color: #0066CC; 
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1); 
    }

    .pagination-controls {
        position: sticky;
        bottom: 0;
        background: #ffffff;
        z-index: 2;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 1rem;
        gap: 0.5rem;
        border-top: 1px solid #eee;
        min-height: 60px;
        border-radius: 20px;
    }

    .page-numbers {
        display: flex;
        gap: 4px;
        align-items: center;
    }

    .page-number, .page-btn {
        min-width: 32px;
        height: 32px;
        padding: 0 6px;
        border: 1px solid #ddd;
        background: #fff;
        border-radius: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #333;
        font-family: 'Poppins';
    }

    .page-number:hover:not(:disabled),
    .page-btn:hover:not(:disabled) {
        background: #f0f0f0;
        border-color: #ccc;
    }

    .page-number.active {
        background: #5932EA;
        color: white;
        border-color: #5932EA;
    }

    .page-ellipsis {
        padding: 0 4px;
        color: #666;
    }

    .page-number:disabled,
    .page-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .hidden-cell {
        display: none;
    }

    .filter-wrapper {
        width: 100%;
        background: transparent;
        border-radius: 20px;
        margin-bottom: 20px;
        min-height: 120px;
        display: flex;
        flex-direction: row;
    }
    
    .reset-wrapper {
        width: auto;
        height: auto;
    }

    .filter-controls {
        padding: 15px;
        border-bottom: 1px solid #eee;
    }

    .filter-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        margin-left: 5px;
        margin-right: 5px;
        width: 90%;
        justify-content: space-between;
    }

    .filter-label {
        font-family: 'Playfair Display';
        font-size: 24px;
        color: white;
        white-space: nowrap;
    }

    .filter-select {
        padding: 6px 12px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-family: 'Poppins';
        font-size: 12px;
        min-width: 120px;
        background: #fff;
    }

    .filter-select:focus {
        outline: none;
        border-color: #5932EA;
        box-shadow: 0 0 0 2px rgba(89, 50, 234, 0.1);
    }

    .reset-button {
        height: 100%;
        background: transparent;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        padding: 0;
    }

    .reset-button span {
        transform: rotate(-90deg);
        white-space: nowrap;
        display: block;
        font-family: 'Playfair Display';
        font-size: 21px;
        letter-spacing: 1px;
    }

    .reset-button:hover {
        background: grey;
    }

    .filtered-text {
        font-family: 'Poppins';
        font-size: 22px;
        font-weight: 600;
        color: black;
    }

    td a {
        text-decoration: underline;
        overflow: hidden; 
        text-overflow: ellipsis; 
        white-space: nowrap; 
        font-family: 'Poppins'; 
        font-size: 14px; 
        color: black;
    }
    
    td a:hover {
        color: grey
    }

    /* Range Slider Styles */
    .range-dropdown {
        position: relative;
        display: inline-block;
    }

    .range-content {
        display: none;
        position: absolute;
        background-color: #fff;
        min-width: 300px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        padding: 20px;
        border-radius: 8px;
        z-index: 1000;
    }

    .range-dropdown:hover .range-content {
        display: block;
    }

    .range-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    .sliders-control {
        position: relative;
        min-height: 50px;
    }

    .form-control {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
        font-family: 'Poppins';
        column-gap: 10px;
    }

    .form-control-container {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .form-control-label {
        font-size: 12px;
        color: #666;
    }

    .form-control-input {
        width: 100px;
        padding: 4px 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 12px;
        font-family: 'Poppins';
    }

    input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        height: 2px;
        width: 100%;
        position: absolute;
        background-color: #C6C6C6;
        pointer-events: none;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        pointer-events: all;
        width: 16px;
        height: 16px;
        background-color: #fff;
        border-radius: 50%;
        box-shadow: 0 0 0 1px #5932EA;
        cursor: pointer;
    }

    input[type="range"]::-moz-range-thumb {
        pointer-events: all;
        width: 16px;
        height: 16px;
        background-color: #fff;
        border-radius: 50%;
        box-shadow: 0 0 0 1px #5932EA;
        cursor: pointer;
    }

    #fromSlider, #goalFromSlider, #raisedFromSlider {
        height: 0;
        z-index: 1;
    }

    .multi-select-dropdown {
        position: relative;
        display: inline-block;
    }

    .multi-select-content {
        display: none;
        position: absolute;
        background-color: #fff;
        min-width: 200px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        padding: 8px;
        border-radius: 8px;
        z-index: 1000;
        max-height: 300px;
        overflow-y: auto;
    }

    .multi-select-dropdown:hover .multi-select-content {
        display: block;
    }

    .multi-select-btn {
        min-width: 150px;
    }

    .category-option {
        padding: 8px 12px;
        cursor: pointer;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Poppins';
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .category-option:hover {
        background-color: #f0f0f0;
    }

    .category-option.selected {
        background-color: #5932EA;
        color: white;
    }

    .category-option[data-value="All Categories"] {
        border-bottom: 1px solid #eee;
        margin-bottom: 8px;
        padding-bottom: 12px;
    }

    .country-option {
        padding: 8px 12px;
        cursor: pointer;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Poppins';
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .country-option:hover {
        background-color: #f0f0f0;
    }

    .country-option.selected {
        background-color: #5932EA;
        color: white;
    }

    .country-option[data-value="All Countries"] {
        border-bottom: 1px solid #eee;
        margin-bottom: 8px;
        padding-bottom: 12px;
    }

    .state-option {
        padding: 8px 12px;
        cursor: pointer;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Poppins';
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .state-option:hover {
        background-color: #f0f0f0;
    }

    .state-option.selected {
        background-color: #5932EA;
        color: white;
    }

    .state-option[data-value="All States"] {
        border-bottom: 1px solid #eee;
        margin-bottom: 8px;
        padding-bottom: 12px;
    }

    .subcategory-option {
        padding: 8px 12px;
        cursor: pointer;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Poppins';
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .subcategory-option:hover {
        background-color: #f0f0f0;
    }

    .subcategory-option.selected {
        background-color: #5932EA;
        color: white;
    }

    .subcategory-option[data-value="All Subcategories"] {
        border-bottom: 1px solid #eee;
        margin-bottom: 8px;
        padding-bottom: 12px;
    }
</style>
"""

# Create table component script with improved search and pagination
script = """
    // Helper functions
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    function createRegexPattern(searchTerm) {
        if (!searchTerm) return null;
        const words = searchTerm.split(/\\s+/).filter(word => word.length > 0);
        const escapedWords = words.map(word => 
            word.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')
        );
        return new RegExp(escapedWords.map(word => `(?=.*${word})`).join(''), 'i');
    }

    function updateTableRows(rows, currentPage, pageSize) {
        const start = (currentPage - 1) * pageSize;
        const end = start + pageSize;
        
        rows.forEach((row, index) => {
            row.style.display = (index >= start && index < end) ? '' : 'none';
        });
    }

    // Add Haversine distance calculation function
    function calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in kilometers
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
            Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    // Optimize distance calculations with a cache
    class DistanceCache {
        constructor() {
            this.userLocation = window.userLocation;
        }

        async initialize() {
            return window.hasLocation;
        }

        getDistance(row) {
            return parseFloat(row.dataset.distance);
        }
    }

    class TableManager {
        constructor() {
            this.searchInput = document.getElementById('table-search');
            this.allRows = Array.from(document.querySelectorAll('#data-table tbody tr'));
            this.visibleRows = this.allRows;
            this.currentPage = 1;
            this.pageSize = 10;
            this.currentSearchTerm = '';
            this.currentFilters = null;
            this.currentSort = 'popularity';
            this.userLocation = window.userLocation;
            this.distanceCache = new DistanceCache();
            this.initialize();
            this.resetFilters();
        }

        // sortRows with async sorting
        async sortRows(sortType) {
            if (sortType === 'popularity') {
                // Sort by popularity score
                this.visibleRows.sort((a, b) => {
                    const scoreA = parseFloat(a.dataset.popularity);
                    const scoreB = parseFloat(b.dataset.popularity);
                    
                    // Handle invalid values
                    if (isNaN(scoreA)) return 1;
                    if (isNaN(scoreB)) return -1;
                    
                    // Sort in descending order
                    return scoreB - scoreA;
                });
            } else if (sortType === 'nearme') {
                if (!this.userLocation) {
                    this.currentSort = 'popularity';
                    document.getElementById('sortFilter').value = 'popularity';
                    this.sortRows('popularity');
                    return;
                }
                
                // Sort only by distance
                this.visibleRows.sort((a, b) => {
                    const distA = parseFloat(a.dataset.distance);
                    const distB = parseFloat(b.dataset.distance);
                    
                    // Handle invalid values
                    if (isNaN(distA)) return 1;
                    if (isNaN(distB)) return -1;
                    
                    return distA - distB;
                });
            } else if (sortType === 'enddate') {
                // Sort by deadline
                this.visibleRows.sort((a, b) => {
                    const deadlineA = new Date(a.dataset.deadline);
                    const deadlineB = new Date(b.dataset.deadline);
                    return deadlineB - deadlineA; 
                });
            } else if (sortType === 'mostfunded') {
                // Sort by pledged amount
                this.visibleRows.sort((a, b) => {
                    const pledgedA = parseFloat(a.dataset.pledged);
                    const pledgedB = parseFloat(b.dataset.pledged);
                    return pledgedB - pledgedA;  // Descending order (most funded first)
                });
            } else if (sortType === 'mostbacked') {
                // Sort by backer count
                this.visibleRows.sort((a, b) => {
                    const backersA = parseInt(a.dataset.backers);
                    const backersB = parseInt(b.dataset.backers);
                    return backersB - backersA;  // Descending order (most backers first)
                });
            } else {
                // Date-based sorting only
                this.visibleRows.sort((a, b) => {
                    const dateA = new Date(a.dataset.date);
                    const dateB = new Date(b.dataset.date);
                    return sortType === 'newest' ? dateB - dateA : dateA - dateB;
                });
            }

            const tbody = document.querySelector('#data-table tbody');
            this.visibleRows.forEach(row => row.parentNode && row.parentNode.removeChild(row));
            this.visibleRows.forEach(row => tbody.appendChild(row));
            
            // Update current page and pagination
            this.currentPage = 1;
            this.updateTable();
        }

        // applyAllFilters with async sorting
        async applyAllFilters() {
            // Start with all rows
            let filteredRows = this.allRows;

            // Apply search if exists
            if (this.currentSearchTerm) {
                const pattern = createRegexPattern(this.currentSearchTerm);
                filteredRows = filteredRows.filter(row => {
                    const text = row.textContent || row.innerText;
                    return pattern.test(text);
                });
            }

            // Apply filters if they exist
            if (this.currentFilters) {
                filteredRows = filteredRows.filter(row => {
                    return this.matchesFilters(row, this.currentFilters);
                });
            }

            // Store filtered results
            this.visibleRows = filteredRows;

            // Apply current sort
            await this.sortRows(this.currentSort);

            // Reset to first page and update display
            this.currentPage = 1;
            this.updateTable();
        }

        // Update applyFilters to handle async
        async applyFilters() {
            // Get all selected categories
            const selectedCategories = Array.from(document.querySelectorAll('.category-option.selected'))
                .map(option => option.dataset.value);

            // Get all selected countries
            const selectedCountries = Array.from(document.querySelectorAll('.country-option.selected'))
                .map(option => option.dataset.value);

            // Get all selected states
            const selectedStates = Array.from(document.querySelectorAll('.state-option.selected'))
                .map(option => option.dataset.value);

            // Get all selected subcategories
            const selectedSubcategories = Array.from(document.querySelectorAll('.subcategory-option.selected'))
                .map(option => option.dataset.value);

            // Collect all current filter values
            this.currentFilters = {
                categories: selectedCategories,
                subcategories: selectedSubcategories,
                countries: selectedCountries,
                states: selectedStates,
                date: document.getElementById('dateFilter').value
            };

            const sortSelect = document.getElementById('sortFilter');
            this.currentSort = sortSelect ? sortSelect.value : 'popularity';
            
            await this.applyAllFilters();
        }

        initialize() {
            this.setupSearchAndPagination();
            this.setupFilters();
            this.setupRangeSlider();
            this.currentSort = 'popularity';  // Set default sort to popularity
            this.applyAllFilters();
            this.updateTable();
        }

        setupSearchAndPagination() {
            // Setup search
            const debouncedSearch = debounce((searchTerm) => {
                this.currentSearchTerm = searchTerm;
                this.applyAllFilters();
            }, 300);

            this.searchInput.addEventListener('input', (e) => {
                debouncedSearch(e.target.value.trim().toLowerCase());
            });

            // Setup pagination controls
            document.getElementById('prev-page').addEventListener('click', () => this.previousPage());
            document.getElementById('next-page').addEventListener('click', () => this.nextPage());
            window.handlePageClick = (page) => this.goToPage(page);
        }

        matchesFilters(row, filters) {
            // Category filter
            const category = row.dataset.category;
            if (!filters.categories.includes('All Categories') && !filters.categories.includes(category)) {
                return false;
            }

            // Subcategory filter
            const subcategory = row.dataset.subcategory;
            if (!filters.subcategories.includes('All Subcategories') && !filters.subcategories.includes(subcategory)) {
                return false;
            }

            // Country filter
            const country = row.querySelector('td:nth-child(5)').textContent.trim();
            if (!filters.countries.includes('All Countries') && !filters.countries.includes(country)) {
                return false;
            }

            // State filter - Extract state from class name instead of text content
            const stateCell = row.querySelector('.state_cell');
            const stateMatch = stateCell ? stateCell.className.match(/state-(\w+)/) : null;
            const state = stateMatch ? stateMatch[1] : '';
            
            if (!filters.states.includes('All States')) {
                const matchingState = filters.states.find(s => 
                    s.toLowerCase() === state.toLowerCase()
                );
                if (!matchingState) return false;
            }

            // Get all other values
            const pledged = parseFloat(row.dataset.pledged);
            const goal = parseFloat(row.dataset.goal);
            const raised = parseFloat(row.dataset.raised);
            const date = new Date(row.dataset.date);

            // Rest of filter checks
            // Check pledged range
            const minPledged = parseFloat(document.getElementById('fromInput').value);
            const maxPledged = parseFloat(document.getElementById('toInput').value);
            if (pledged < minPledged || pledged > maxPledged) return false;

            // Check goal range
            const minGoal = parseFloat(document.getElementById('goalFromInput').value);
            const maxGoal = parseFloat(document.getElementById('goalToInput').value);
            if (goal < minGoal || goal > maxGoal) return false;

            // Check raised range
            const minRaised = parseFloat(document.getElementById('raisedFromInput').value);
            const maxRaised = parseFloat(document.getElementById('raisedToInput').value);
            const raisedValue = parseFloat(row.dataset.raised);
            
            // Handle the case where raised is exactly 0%
            if (raisedValue === 0 && minRaised > 0) return false;
            if (raisedValue < minRaised || raisedValue > maxRaised) return false;

            // Date filter
            if (filters.date !== 'All Time') {
                const now = new Date();
                let compareDate = new Date();
                
                switch(filters.date) {
                    case 'Last Month': compareDate.setMonth(now.getMonth() - 1); break;
                    case 'Last 6 Months': compareDate.setMonth(now.getMonth() - 6); break;
                    case 'Last Year': compareDate.setFullYear(now.getFullYear() - 1); break;
                    case 'Last 5 Years': compareDate.setFullYear(now.getFullYear() - 5); break;
                    case 'Last 10 Years': compareDate.setFullYear(now.getFullYear() - 10); break;
                }
                
                if (date < compareDate) return false;
            }

            return true;
        }

        resetFilters() {
            // Reset category selections
            const categoryOptions = document.querySelectorAll('.category-option');
            categoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCategoriesOption = document.querySelector('.category-option[data-value="All Categories"]');
            allCategoriesOption.classList.add('selected');
            const categoryBtn = document.querySelector('.multi-select-btn');
            categoryBtn.textContent = 'All Categories';

            // Reset country selections
            const countryOptions = document.querySelectorAll('.country-option');
            countryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCountriesOption = document.querySelector('.country-option[data-value="All Countries"]');
            allCountriesOption.classList.add('selected');
            const countryBtn = countryOptions[0].closest('.multi-select-dropdown').querySelector('.multi-select-btn');
            countryBtn.textContent = 'All Countries';

            // Reset state selections
            const stateOptions = document.querySelectorAll('.state-option');
            stateOptions.forEach(opt => opt.classList.remove('selected'));
            const allStatesOption = document.querySelector('.state-option[data-value="All States"]');
            allStatesOption.classList.add('selected');
            const stateBtn = stateOptions[0].closest('.multi-select-dropdown').querySelector('.multi-select-btn');
            stateBtn.textContent = 'All States';

            // Reset subcategory selections
            const subcategoryOptions = document.querySelectorAll('.subcategory-option');
            subcategoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allSubcategoriesOption = document.querySelector('.subcategory-option[data-value="All Subcategories"]');
            allSubcategoriesOption.classList.add('selected');
            const subcategoryBtn = subcategoryOptions[0].closest('.multi-select-dropdown').querySelector('.multi-select-btn');
            subcategoryBtn.textContent = 'All Subcategories';

            // Reset the stored selections in the Sets
            if (window.selectedCategories) window.selectedCategories.clear();
            if (window.selectedCountries) window.selectedCountries.clear();
            if (window.selectedStates) window.selectedStates.clear();
            if (window.selectedSubcategories) window.selectedSubcategories.clear();

            // Re-add "All" options to the Sets
            if (window.selectedCategories) window.selectedCategories.add('All Categories');
            if (window.selectedCountries) window.selectedCountries.add('All Countries');
            if (window.selectedStates) window.selectedStates.add('All States');
            if (window.selectedSubcategories) window.selectedSubcategories.add('All Subcategories');

            // Reset all range sliders and inputs
            if (this.rangeSliderElements) {
                const { 
                    fromSlider, toSlider, fromInput, toInput,
                    goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                    raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                    fillSlider 
                } = this.rangeSliderElements;

                // Reset pledged amount range
                fromSlider.value = fromSlider.min;
                toSlider.value = toSlider.max;
                fromInput.value = fromSlider.min;
                toInput.value = toSlider.max;
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);

                // Reset goal amount range
                goalFromSlider.value = goalFromSlider.min;
                goalToSlider.value = goalToSlider.max;
                goalFromInput.value = goalFromSlider.min;
                goalToInput.value = goalToSlider.max;
                fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);

                // Reset percentage raised range
                raisedFromSlider.value = raisedFromSlider.min;
                raisedToSlider.value = raisedToSlider.max;
                raisedFromInput.value = raisedFromSlider.min;
                raisedToInput.value = raisedToSlider.max;
                fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
            }

            this.searchInput.value = '';
            this.currentSearchTerm = '';
            this.currentFilters = null;
            this.currentSort = 'popularity';
            this.visibleRows = this.allRows;
            this.applyAllFilters();
        }

        updateTable() {
            // Hide all rows first
            this.allRows.forEach(row => row.style.display = 'none');
            
            // Calculate visible range
            const start = (this.currentPage - 1) * this.pageSize;
            const end = Math.min(start + this.pageSize, this.visibleRows.length);
            
            // Show only rows for current page
            this.visibleRows.slice(start, end).forEach(row => {
                row.style.display = '';
            });

            this.updatePagination();
            this.adjustHeight();
        }

        updatePagination() {
            const totalPages = Math.max(1, Math.ceil(this.visibleRows.length / this.pageSize));
            const pageNumbers = this.generatePageNumbers(totalPages);
            const container = document.getElementById('page-numbers');
            
            container.innerHTML = pageNumbers.map(page => {
                if (page === '...') {
                    return '<span class="page-ellipsis">...</span>';
                }
                return `<button class="page-number ${page === this.currentPage ? 'active' : ''}"
                    ${page === this.currentPage ? 'disabled' : ''} 
                    onclick="handlePageClick(${page})">${page}</button>`;
            }).join('');

            document.getElementById('prev-page').disabled = this.currentPage <= 1;
            document.getElementById('next-page').disabled = this.currentPage >= totalPages;
        }

        generatePageNumbers(totalPages) {
            let pages = [];
            if (totalPages <= 10) {
                pages = Array.from({length: totalPages}, (_, i) => i + 1);
            } else {
                if (this.currentPage <= 7) {
                    pages = [...Array.from({length: 7}, (_, i) => i + 1), '...', totalPages - 1, totalPages];
                } else if (this.currentPage >= totalPages - 6) {
                    pages = [1, 2, '...', ...Array.from({length: 7}, (_, i) => totalPages - 6 + i)];
                } else {
                    pages = [1, 2, '...', this.currentPage - 1, this.currentPage, this.currentPage + 1, '...', totalPages - 1, totalPages];
                }
            }
            return pages;
        }

        previousPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.updateTable();
            }
        }

        nextPage() {
            const totalPages = Math.ceil(this.visibleRows.length / this.pageSize);
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.updateTable();
            }
        }

        goToPage(page) {
            const totalPages = Math.ceil(this.visibleRows.length / this.pageSize);
            if (page >= 1 && page <= totalPages) {
                this.currentPage = page;
                this.updateTable();
            }
        }

        adjustHeight() {
            requestAnimationFrame(() => {
                const elements = {
                    titleWrapper: document.querySelector('.title-wrapper'),
                    filterWrapper: document.querySelector('.filter-wrapper'),
                    tableWrapper: document.querySelector('.table-wrapper'),
                    tableContainer: document.querySelector('.table-container'),
                    table: document.querySelector('#data-table'),
                    controls: document.querySelector('.table-controls'),
                    pagination: document.querySelector('.pagination-controls')
                };

                if (!Object.values(elements).every(el => el)) return;

                // Count visible rows in current page
                const visibleRowCount = this.visibleRows.slice(
                    (this.currentPage - 1) * this.pageSize,
                    this.currentPage * this.pageSize
                ).length;

                // Constants
                const rowHeight = 52;        // Height per row including padding
                const headerHeight = 60;     // Table header height
                const controlsHeight = elements.controls.offsetHeight;
                const paginationHeight = elements.pagination.offsetHeight;
                const padding = 40;
                const minTableHeight = 400;  // Minimum table content height

                // Calculate table content height
                const tableContentHeight = (visibleRowCount * rowHeight) + headerHeight;
                const actualTableHeight = Math.max(tableContentHeight, minTableHeight);

                // Set dimensions
                elements.tableContainer.style.height = `${actualTableHeight}px`;
                elements.tableWrapper.style.height = `${actualTableHeight + controlsHeight + paginationHeight}px`;

                // Calculate final component height
                const finalHeight = 
                    elements.titleWrapper.offsetHeight +
                    elements.filterWrapper.offsetHeight +
                    actualTableHeight +
                    controlsHeight +
                    paginationHeight +
                    padding;

                // Update Streamlit frame height if changed significantly
                if (!this.lastHeight || Math.abs(this.lastHeight - finalHeight) > 10) {
                    this.lastHeight = finalHeight;
                    Streamlit.setFrameHeight(finalHeight);
                }
            });
        }

        setupFilters() {
            // Initialize global Sets to track selections
            window.selectedCategories = new Set(['All Categories']);
            window.selectedCountries = new Set(['All Countries']);
            window.selectedStates = new Set(['All States']);
            window.selectedSubcategories = new Set(['All Subcategories']);

            // Get button elements by ID
            const categoryBtn = document.getElementById('categoryFilterBtn');
            const countryBtn = document.getElementById('countryFilterBtn');
            const stateBtn = document.getElementById('stateFilterBtn');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            // Create a single update function for all buttons
            const updateButtonText = (selectedItems, buttonElement) => {
                if (!buttonElement) return;
                
                const selectedArray = Array.from(selectedItems);
                if (selectedArray[0] && selectedArray[0].startsWith('All')) {
                    buttonElement.textContent = selectedArray[0];
                } else {
                    const sortedArray = selectedArray.sort((a, b) => a.localeCompare(b));
                    if (sortedArray.length > 2) {
                        buttonElement.textContent = `${sortedArray[0]}, ${sortedArray[1]} +${sortedArray.length - 2}`;
                    } else {
                        buttonElement.textContent = sortedArray.join(', ');
                    }
                }
            };

            // Setup multi-select handlers
            const setupMultiSelect = (options, selectedSet, allValue, buttonElement) => {
                const allOption = document.querySelector(`[data-value="${allValue}"]`);
                
                options.forEach(option => {
                    option.addEventListener('click', (e) => {
                        const clickedValue = e.target.dataset.value;
                        
                        if (clickedValue === allValue) {
                            options.forEach(opt => opt.classList.remove('selected'));
                            selectedSet.clear();
                            selectedSet.add(allValue);
                            allOption.classList.add('selected');
                        } else {
                            allOption.classList.remove('selected');
                            selectedSet.delete(allValue);
                            
                            e.target.classList.toggle('selected');
                            if (e.target.classList.contains('selected')) {
                                selectedSet.add(clickedValue);
                            } else {
                                selectedSet.delete(clickedValue);
                            }
                            
                            if (selectedSet.size === 0) {
                                allOption.classList.add('selected');
                                selectedSet.add(allValue);
                            }
                        }
                        
                        updateButtonText(selectedSet, buttonElement);
                        this.applyFilters();
                    });
                });

                // Initialize button text
                updateButtonText(selectedSet, buttonElement);
            };

            // Setup each multi-select with the correct button element
            setupMultiSelect(
                document.querySelectorAll('.category-option'),
                window.selectedCategories,
                'All Categories',
                categoryBtn
            );

            setupMultiSelect(
                document.querySelectorAll('.country-option'),
                window.selectedCountries,
                'All Countries',
                countryBtn
            );

            setupMultiSelect(
                document.querySelectorAll('.state-option'),
                window.selectedStates,
                'All States',
                stateBtn
            );

            setupMultiSelect(
                document.querySelectorAll('.subcategory-option'),
                window.selectedSubcategories,
                'All Subcategories',
                subcategoryBtn
            );

            // Setup other filters
            const filterIds = ['dateFilter', 'sortFilter'];
            filterIds.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.addEventListener('change', () => this.applyFilters());
                }
            });

            // Add reset button handler
            const resetButton = document.getElementById('resetFilters');
            if (resetButton) {
                resetButton.addEventListener('click', () => this.resetFilters());
            }

            // Add range slider initialization
            this.setupRangeSlider();
        }

        setupRangeSlider() {
            // Setup for pledged amount slider
            const fromSlider = document.getElementById('fromSlider');
            const toSlider = document.getElementById('toSlider');
            const fromInput = document.getElementById('fromInput');
            const toInput = document.getElementById('toInput');

            // Setup for goal amount slider
            const goalFromSlider = document.getElementById('goalFromSlider');
            const goalToSlider = document.getElementById('goalToSlider');
            const goalFromInput = document.getElementById('goalFromInput');
            const goalToInput = document.getElementById('goalToInput');

            // Setup for percentage raised slider
            const raisedFromSlider = document.getElementById('raisedFromSlider');
            const raisedToSlider = document.getElementById('raisedToSlider');
            const raisedFromInput = document.getElementById('raisedFromInput');
            const raisedToInput = document.getElementById('raisedToInput');

            let inputTimeout;

            const fillSlider = (from, to, sliderColor, rangeColor, controlSlider) => {
                const rangeDistance = controlSlider.max - controlSlider.min;
                const fromPosition = from.value - controlSlider.min;
                const toPosition = to.value - controlSlider.min;
                controlSlider.style.background = `linear-gradient(
                    to right,
                    ${sliderColor} 0%,
                    ${sliderColor} ${(fromPosition)/(rangeDistance)*100}%,
                    ${rangeColor} ${((fromPosition)/(rangeDistance))*100}%,
                    ${rangeColor} ${(toPosition)/(rangeDistance)*100}%, 
                    ${sliderColor} ${(toPosition)/(rangeDistance)*100}%, 
                    ${sliderColor} 100%)`;
            }

            const debouncedApplyFilters = debounce(() => this.applyFilters(), 100);

            const controlFromSlider = (fromSlider, toSlider, fromInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                if (from > to) {
                    fromSlider.value = to;
                    fromInput.value = to;
                } else {
                    fromInput.value = from;
                }
            };

            const controlToSlider = (fromSlider, toSlider, toInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                if (from <= to) {
                    toSlider.value = to;
                    toInput.value = to;
                } else {
                    toInput.value = from;
                    toSlider.value = from;
                }
            };

            const getParsedValue = (fromSlider, toSlider) => {
                const from = parseInt(fromSlider.value);
                const to = parseInt(toSlider.value);
                return [from, to];
            };

            const validateAndUpdateRange = (input, isMin = true, immediate = false) => {
                const updateValues = () => {
                    let value = parseInt(input.value);
                    const minAllowed = parseInt(input.min);
                    const maxAllowed = parseInt(input.max);
                    const isGoalInput = input.id.startsWith('goal');
                    
                    if (isNaN(value)) {
                        value = isMin ? minAllowed : maxAllowed;
                    }
                    
                    const fromSlider = isGoalInput ? goalFromSlider : this.rangeSliderElements.fromSlider;
                    const toSlider = isGoalInput ? goalToSlider : this.rangeSliderElements.toSlider;
                    
                    if (isMin) {
                        const maxValue = parseInt(toSlider.value);
                        value = Math.max(minAllowed, Math.min(maxValue, value));
                        fromSlider.value = value;
                        input.value = value;
                    } else {
                        const minValue = parseInt(fromSlider.value);
                        value = Math.max(minValue, Math.min(maxAllowed, value));
                        toSlider.value = value;
                        input.value = value;
                    }
                    
                    fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                    debouncedApplyFilters();
                };

                if (immediate) {
                    clearTimeout(inputTimeout);
                    updateValues();
                } else {
                    clearTimeout(inputTimeout);
                    inputTimeout = setTimeout(updateValues, 1000);
                }
            };

            // Event listeners for pledged amount slider
            fromSlider.addEventListener('input', (e) => {
                controlFromSlider(fromSlider, toSlider, fromInput);
                debouncedApplyFilters();
            });

            toSlider.addEventListener('input', (e) => {
                controlToSlider(fromSlider, toSlider, toInput);
                debouncedApplyFilters();
            });

            // Event listeners for goal amount slider
            goalFromSlider.addEventListener('input', (e) => {
                controlFromSlider(goalFromSlider, goalToSlider, goalFromInput);
                debouncedApplyFilters();
            });

            goalToSlider.addEventListener('input', (e) => {
                controlToSlider(goalFromSlider, goalToSlider, goalToInput);
                debouncedApplyFilters();
            });

            // Add event listeners for percentage raised slider
            raisedFromSlider.addEventListener('input', (e) => {
                controlFromSlider(raisedFromSlider, raisedToSlider, raisedFromInput);
                debouncedApplyFilters();
            });

            raisedToSlider.addEventListener('input', (e) => {
                controlToSlider(raisedFromSlider, raisedToSlider, raisedToInput);
                debouncedApplyFilters();
            });

            // Input handlers for both sliders
            [fromInput, goalFromInput, raisedFromInput].forEach(input => {
                input.addEventListener('input', () => {
                    validateAndUpdateRange(input, true, false);
                });
            });

            [toInput, goalToInput, raisedToInput].forEach(input => {
                input.addEventListener('input', () => {
                    validateAndUpdateRange(input, false, false);
                });
            });

            // Add key events for immediate validation on Enter
            fromInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(fromInput, true, true);
                }
            });

            toInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(toInput, false, true);
                }
            });

            // Add key events for goal slider inputs
            goalFromInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(goalFromInput, true, true);
                }
            });

            goalToInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(goalToInput, false, true);
                }
            });

            // Add key events for percentage raised slider inputs
            raisedFromInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(raisedFromInput, true, true);
                }
            });

            raisedToInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    validateAndUpdateRange(raisedToInput, false, true);
                }
            });

            // Also handle blur events for immediate validation
            fromInput.addEventListener('blur', () => {
                validateAndUpdateRange(fromInput, true, true);
            });

            toInput.addEventListener('blur', () => {
                validateAndUpdateRange(toInput, false, true);
            });

            goalFromInput.addEventListener('blur', () => {
                validateAndUpdateRange(goalFromInput, true, true);
            });

            goalToInput.addEventListener('blur', () => {
                validateAndUpdateRange(goalToInput, false, true);
            });

            raisedFromInput.addEventListener('blur', () => {
                validateAndUpdateRange(raisedFromInput, true, true);
            });

            raisedToInput.addEventListener('blur', () => {
                validateAndUpdateRange(raisedToInput, false, true);
            });

            // Store references for reset function
            this.rangeSliderElements = {
                fromSlider, toSlider, fromInput, toInput,
                goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                fillSlider
            };

            // Initial setup
            fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
            fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);
            fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
        }
    }

    function onRender(event) {
        if (!window.rendered) {
            window.tableManager = new TableManager();
            window.rendered = true;
            
            // Add resize observer to handle dynamic content changes
            const resizeObserver = new ResizeObserver(() => {
                window.tableManager.adjustHeight();
            });
            resizeObserver.observe(document.querySelector('.table-wrapper'));
        }
    }

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
"""

# Create and use the component
table_component = generate_component('searchable_table', template=css + template, script=script)
table_component()

# st.dataframe(df) # Display the dataframe