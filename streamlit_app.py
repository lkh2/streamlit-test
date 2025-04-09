import time
import streamlit as st
import streamlit.components.v1 as components
import tempfile, os
import pandas as pd
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

@st.cache_data
def load_data_from_parquet_chunks() -> pl.LazyFrame: # Return LazyFrame
    """
    Scan data from compressed parquet chunks lazily.
    Combines chunks first, then scans the resulting file.
    """
    chunk_files = glob.glob("parquet_gz_chunks/*.part")

    if not chunk_files:
        st.error("No parquet chunks found in parquet_gz_chunks folder. Please run database_download.py first.")
        # Return an empty LazyFrame by scanning a non-existent file? Or handle differently?
        # For now, let's return an empty eager DF and handle it below.
        # A truly empty LF is harder to signal errors with downstream.
        # Returning empty eager DF to be checked easily later.
        return pl.DataFrame().lazy()

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Found {len(chunk_files)} chunk files. Combining chunks...")

    combined_filename = None
    decompressed_filename = None
    lf = None # Initialize LazyFrame variable

    try:
        # Create a temporary file to store the combined chunks
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet.gz') as combined_file:
            combined_filename = combined_file.name
            chunk_files = sorted(chunk_files)
            for i, chunk_file in enumerate(chunk_files):
                try:
                    with open(chunk_file, 'rb') as f:
                        combined_file.write(f.read())
                    progress_bar.progress((i + 1) / (2 * len(chunk_files)))
                    status_text.text(f"Combined chunk {i+1}/{len(chunk_files)}")
                except Exception as e:
                    st.warning(f"Error reading chunk {chunk_file}: {str(e)}")

        status_text.text("Decompressing combined file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as decompressed_file:
            decompressed_filename = decompressed_file.name
            with gzip.open(combined_filename, 'rb') as gz_file:
                decompressed_file.write(gz_file.read())

        # Scan the decompressed parquet file using Polars Lazily
        status_text.text("Scanning pre-processed parquet data...")
        progress_bar.progress(0.75)

        # Scan the parquet file lazily
        lf = pl.scan_parquet(decompressed_filename)

        # Optional: Fetch a small amount to check if scan worked without loading all
        # try:
        #     _ = lf.fetch(1) # Check if scan is valid
        # except Exception as scan_error:
        #      st.error(f"Error scanning parquet file: {scan_error}")
        #      return pl.DataFrame().lazy() # Return empty lazy frame on scan error

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # We don't know the length until collect, maybe fetch()?
        # st.success(f"Successfully scanned projects!") # Cannot show count yet
        return lf

    except Exception as e:
        st.error(f"Error processing parquet file for scanning: {str(e)}")
        return pl.DataFrame().lazy() # Return empty LazyFrame on error
    finally:
        # Clean up temporary files
        # Note: Deleting decompressed_filename immediately might cause issues
        # if the scan hasn't fully completed its metadata read.
        # Consider delaying cleanup or handling it more robustly if needed.
        try:
            if combined_filename and os.path.exists(combined_filename):
                os.unlink(combined_filename)
            # Keep decompressed file around until the end? Or manage lifetime?
            # For simplicity now, we risk deleting it early, but scan_parquet likely reads metadata quickly.
            if decompressed_filename and os.path.exists(decompressed_filename):
                 # Pass the filename to be deleted later if needed
                 # For now, delete it, assuming scan_parquet has read metadata
                 os.unlink(decompressed_filename)
                 pass # Or manage cleanup later
        except Exception as e:
            st.warning(f"Error cleaning up temporary files: {str(e)}")


# Load pre-processed data lazily
lf = load_data_from_parquet_chunks()

# --- Check if LazyFrame is potentially empty ---
# Fetching 1 row is a relatively cheap way to check if the scan is valid
# and if there's any data, without loading everything.
try:
    # Use fetch instead of collect for minimal data loading
    schema_check = lf.fetch(0) # Fetch 0 rows just to validate schema and connectivity
    if schema_check.is_empty() and schema_check.width == 0 : # Check if schema is also empty
         st.error("Failed to load data or data file is empty. Please check logs and ensure database_download.py ran successfully.")
         st.stop()
    # We can now assume the LazyFrame is likely valid and has columns
    print("LazyFrame Schema:", lf.schema)

except Exception as e:
     st.error(f"Error during initial data check: {e}. Cannot proceed.")
     st.stop()


# Apply styling to State column (Lazy)
# Check if 'State' exists in the schema
if 'State' in lf.schema:
    lf = lf.with_columns(
        (
            pl.lit('<div class="state_cell state-')
            + pl.col('State').str.to_lowercase().fill_null('unknown')
            + pl.lit('">')
            + pl.col('State').str.to_lowercase().fill_null('unknown')
            + pl.lit('</div>')
        ).alias('State')
    )
else:
    st.warning("Column 'State' not found in the schema. Skipping state styling.")

# Add country coordinates by joining country.csv (Lazy)
@st.cache_data
def load_country_data() -> pl.DataFrame: # Keep this eager, it's small
    try:
        # Use read_csv for eager loading of the small country file
        country_df = pl.read_csv('country.csv')
        country_df = country_df.select(['country', 'latitude', 'longitude']).rename({'latitude': 'country_lat', 'longitude': 'country_lon'})
        return country_df
    except Exception as e:
        st.error(f"Failed to load country.csv: {e}")
        return pl.DataFrame() # Return empty eager DF

country_data = load_country_data()

# Join with country_data (Lazy) and create latitude/longitude columns
if not country_data.is_empty() and 'Country Code' in lf.schema:
     # Convert small country_data to LazyFrame for the join
     lf = lf.join(country_data.lazy(),
                  left_on='Country Code',
                  right_on='country',
                  how='left')
     lf = lf.with_columns([
          pl.col('country_lat').fill_null(0.0).alias('latitude'),
          pl.col('country_lon').fill_null(0.0).alias('longitude')
     ])
     # Drop columns lazily - check schema first
     cols_to_drop_after_join = [col for col in ['country_lat', 'country_lon'] if col in lf.schema]
     if cols_to_drop_after_join:
          lf = lf.drop(cols_to_drop_after_join)
else:
     st.warning("Could not join country data or 'Country Code' column missing in schema. Creating default Latitude/Longitude columns (0.0).")
     # Add columns lazily
     lf = lf.with_columns([
         pl.lit(0.0).cast(pl.Float64).alias('latitude'),
         pl.lit(0.0).cast(pl.Float64).alias('longitude')
     ])

# --- RE-ADD Geolocation Fetching ---
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

# --- RE-ADD Distance Calculation Function and Logic (Lazy) ---
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers

    # Ensure inputs are floats before conversion
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        return float('inf') # Return infinity if conversion fails

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Calculate distances if user location is available (Lazy)
if user_location and 'latitude' in lf.schema and 'longitude' in lf.schema:
    print("Adding distance calculation to LazyFrame plan...")
    user_lat = float(user_location['latitude'])
    user_lon = float(user_location['longitude'])

    # Apply the distance function lazily using Polars expressions
    lf = lf.with_columns(
        pl.struct(['latitude', 'longitude'])
        # The apply function here will be executed when .collect() is called
        .apply(lambda x: calculate_distance(user_lat, user_lon, x['latitude'], x['longitude']),
               return_dtype=pl.Float64) # Specify return type for apply in lazy mode
        .alias('Distance')
    )
    print("Distance calculation added to plan.")

else:
    print("User location not available or lat/lon columns missing in schema. Setting Distance to infinity (Lazy).")
    # Add Distance column lazily
    lf = lf.with_columns(pl.lit(float('inf')).cast(pl.Float64).alias('Distance'))

# --- Calculate Filter Options and Min/Max (More Efficiently) ---

# Prepare filter options (operate lazily as much as possible)
@st.cache_data
def get_filter_options(_lf: pl.LazyFrame): # Pass LazyFrame, use _ prefix to indicate mutation
    print("Calculating filter options...")
    # Collect unique values only for required columns
    options = {
        'categories': ['All Categories'],
        'subcategories': ['All Subcategories'],
        'countries': ['All Countries'],
        'states': ['All States'],
        'date_ranges': [
            'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
            'Last 5 Years', 'Last 10 Years'
        ]
    }

    try:
        if 'Category' in _lf.schema:
            categories_unique = _lf.select(pl.col('Category')).unique().collect()['Category']
            options['categories'] += sorted(categories_unique.filter(categories_unique.is_not_null() & (categories_unique != "N/A")).to_list())

        if 'Subcategory' in _lf.schema:
             subcategories_unique = _lf.select(pl.col('Subcategory')).unique().collect()['Subcategory']
             options['subcategories'] += sorted(subcategories_unique.filter(subcategories_unique.is_not_null() & (subcategories_unique != "N/A")).to_list())

        if 'Country' in _lf.schema:
             countries_unique = _lf.select(pl.col('Country')).unique().collect()['Country']
             options['countries'] += sorted(countries_unique.filter(countries_unique.is_not_null() & (countries_unique != "N/A")).to_list())

        # State extraction needs collection first, as str.extract is complex
        if 'State' in _lf.schema and _lf.schema['State'] == pl.Utf8:
             # Collect only the State column for processing
             states_collected = _lf.select('State').collect()['State']
             # Check if the column likely contains the HTML structure on the collected data
             sample_state = states_collected.head(1).to_list()
             if sample_state and sample_state[0] and sample_state[0].startswith('<div class="state_cell state-'):
                  # Perform extraction on the collected Series
                  extracted_states = states_collected.str.extract(r'state-(\w+)', 1).unique().drop_nulls().to_list()
                  options['states'] += sorted([state.capitalize() for state in extracted_states if state != 'unknown'])
             else: # Assume it's plain text if no HTML structure detected
                  plain_states = states_collected.filter(states_collected.is_not_null() & (states_collected != "N/A")).unique().to_list()
                  options['states'] += sorted([s.capitalize() for s in plain_states])
        print("Filter options calculated.")

    except Exception as e:
         st.error(f"Error calculating filter options: {e}")
         # Return default options on error
         options = {k: v[:1] for k, v in options.items()} # Keep only "All..." options
         options['date_ranges'] = [ # Restore date ranges
            'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
            'Last 5 Years', 'Last 10 Years'
         ]
    return options

# Calculate filter options using the LazyFrame
filter_options = get_filter_options(lf)

# Calculate Min/Max values lazily
min_pledged, max_pledged = 0, 1000
min_goal, max_goal = 0, 10000
min_raised, max_raised = 0, 500 # Default raised range
max_raised_display_cap = 5000 # Cap for slider usability

required_minmax_cols = ['Raw Pledged', 'Raw Goal', 'Raw Raised']
if all(col in lf.schema for col in required_minmax_cols):
    print("Calculating min/max filter ranges...")
    try:
        # Compute min/max in a single collect call if possible
        min_max_vals = lf.select([
            pl.min('Raw Pledged').alias('min_pledged'),
            pl.max('Raw Pledged').alias('max_pledged'),
            pl.min('Raw Goal').alias('min_goal'),
            pl.max('Raw Goal').alias('max_goal'),
            pl.min('Raw Raised').alias('min_raised'),
            pl.max('Raw Raised').alias('max_raised_calc')
        ]).collect()

        min_pledged = int(min_max_vals['min_pledged'][0]) if min_max_vals['min_pledged'][0] is not None else 0
        max_pledged = int(min_max_vals['max_pledged'][0]) if min_max_vals['max_pledged'][0] is not None else 1000
        min_goal = int(min_max_vals['min_goal'][0]) if min_max_vals['min_goal'][0] is not None else 0
        max_goal = int(min_max_vals['max_goal'][0]) if min_max_vals['max_goal'][0] is not None else 10000
        min_raised = int(min_max_vals['min_raised'][0]) if min_max_vals['min_raised'][0] is not None else 0
        max_raised_calc_val = min_max_vals['max_raised_calc'][0]
        # Cap max_raised for the slider range
        max_raised = int(max_raised_calc_val) if max_raised_calc_val is not None and max_raised_calc_val < max_raised_display_cap else max_raised_display_cap
        print("Min/max ranges calculated.")
    except Exception as e:
        st.error(f"Error calculating min/max filter ranges: {e}. Using defaults.")
        # Defaults are already set
else:
    st.error("Missing columns required for min/max filter ranges in schema. Using defaults.")
    # Defaults are already set


# --- Collect Data ONLY before generating HTML ---
print("Collecting final DataFrame for display...")
start_collect_time = time.time()
try:
    df_collected = lf.collect(streaming=True) # Use streaming for potential memory benefits
    collect_duration = time.time() - start_collect_time
    st.success(f"Loaded {len(df_collected)} projects successfully! (Collection took {collect_duration:.2f}s)")
except Exception as e:
    st.error(f"Error collecting final DataFrame: {e}")
    df_collected = pl.DataFrame() # Ensure df_collected exists but is empty
    st.stop()

if df_collected.is_empty():
     st.error("Data collection resulted in an empty DataFrame. Cannot display table.")
     st.stop()


# --- Generate Table HTML (using collected DataFrame) ---
def generate_table_html(df_display: pl.DataFrame): # Expect Eager DataFrame
    # Define visible columns
    visible_columns = ['Project Name', 'Creator', 'Pledged Amount', 'Link', 'Country', 'State']

    # Ensure all required columns for data attributes exist (ADD Distance back)
    required_data_cols = [
        'Category', 'Subcategory', 'Raw Pledged', 'Raw Goal', 'Raw Raised',
        'Raw Date', 'Raw Deadline', 'Backer Count', 'latitude', 'longitude',
        'Country Code', 'Distance', 'Popularity Score' # ADDED Distance
    ]
    missing_data_cols = [col for col in required_data_cols if col not in df_display.columns]
    if missing_data_cols:
        st.error(f"Missing required columns for table generation: {missing_data_cols}. Check data processing steps.")
        return "", ""

    # Ensure visible columns exist
    missing_visible_cols = [col for col in visible_columns if col not in df_display.columns]
    if missing_visible_cols:
         st.error(f"Missing visible columns for table: {missing_visible_cols}. Check database_download.py.")
         # Attempt to continue with available columns
         visible_columns = [col for col in visible_columns if col in df_display.columns]
         if not visible_columns:
              return "", ""


    # Generate header for visible columns only
    header_html = ''.join(f'<th scope="col">{column}</th>' for column in visible_columns)

    # Generate table rows with raw values in data attributes
    rows_html = ''
    # Convert collected DataFrame to dicts efficiently
    data_dicts = df_display.to_dicts()

    for row in data_dicts:
         # Safely format data attributes (ADD Distance back)
        data_attrs = f'''
            data-category="{row.get('Category', 'N/A')}"
            data-subcategory="{row.get('Subcategory', 'N/A')}"
            data-pledged="{row.get('Raw Pledged', 0.0):.2f}"
            data-goal="{row.get('Raw Goal', 0.0):.2f}"
            data-raised="{row.get('Raw Raised', 0.0):.2f}"
            data-date="{row.get('Raw Date').strftime('%Y-%m-%d') if row.get('Raw Date') else 'N/A'}"
            data-deadline="{row.get('Raw Deadline').strftime('%Y-%m-%d') if row.get('Raw Deadline') else 'N/A'}"
            data-backers="{row.get('Backer Count', 0)}"
            data-latitude="{row.get('latitude', 0.0):.6f}"
            data-longitude="{row.get('longitude', 0.0):.6f}"
            data-country-code="{row.get('Country Code', 'N/A')}"
            data-distance="{row.get('Distance', float('inf')):.2f}"
            data-popularity="{row.get('Popularity Score', 0.0):.6f}"
        '''

        # Create visible cells with special handling for Link column
        visible_cells = ''
        for col in visible_columns:
            value = row.get(col, 'N/A') # Default value if column somehow missing in dict
            if col == 'Link':
                url = str(value) if value else '#'
                # Truncate long URLs for display if needed
                display_url = url if len(url) < 60 else url[:57] + '...'
                visible_cells += f'<td><a href="{url}" target="_blank" title="{url}">{display_url}</a></td>'
            elif col == 'State': # State column already contains HTML
                 visible_cells += f'<td>{value}</td>'
            else:
                # Escape potential HTML in other cells
                import html
                visible_cells += f'<td>{html.escape(str(value))}</td>'

        rows_html += f'<tr class="table-row" {data_attrs}>{visible_cells}</tr>'

    return header_html, rows_html

# Generate HTML using the collected DataFrame
header_html, rows_html = generate_table_html(df_collected)


# --- TEMPLATE AND CSS definitions ---
# RE-ADD 'Near Me' sort option conditionally
# RE-ADD userLocation to script tag
# --- Check user_location BEFORE generating the template ---
near_me_option_html = '<option value="nearme">Near Me</option>' if user_location else '<option value="nearme" disabled>Near Me (Location needed)</option>'

template = f"""
<script>
    // RE-ADD user location to JavaScript
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
                {near_me_option_html}
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
                            <input id="goalFromSlider" type="range" value="{min_goal}" min="{min_goal}" max="{max_goal}"/>
                            <input id="goalToSlider" type="range" value="{max_goal}" min="{min_goal}" max="{max_goal}"/>
                        </div>
                        <div class="form-control">
                            <div class="form-control-container">
                                <span class="form-control-label">Min $</span>
                                <input class="form-control-input" type="number" id="goalFromInput" value="{min_goal}" min="{min_goal}" max="{max_goal}"/>
                            </div>
                            <div class="form-control-container">
                                <span class="form-control-label">Max $</span>
                                <input class="form-control-input" type="number" id="goalToInput" value="{max_goal}" min="{min_goal}" max="{max_goal}"/>
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
"""

# CSS remains the same
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

# --- SCRIPT definition ---
# No changes needed in the script as it operates on the rendered HTML/data attributes
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

    // RE-ADD Haversine distance function (or confirm it exists)
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

    // RE-ADD DistanceCache class (or confirm it exists)
    class DistanceCache {
        constructor() {
            this.userLocation = window.userLocation;
        }

        async initialize() {
            return window.hasLocation;
        }

        getDistance(row) {
            // Assuming distance is pre-calculated and stored in data-distance
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

        // sortRows - RE-ADD 'nearme' case
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
            } else if (sortType === 'nearme') { // RE-ADD case
                if (!this.userLocation) {
                    console.warn("Attempted to sort by 'Near Me' without location. Falling back to popularity.");
                    // Reset sort dropdown and internal state
                    const sortSelect = document.getElementById('sortFilter');
                     if (sortSelect.value === 'nearme') {
                           sortSelect.value = 'popularity';
                           this.currentSort = 'popularity';
                           // Maybe disable the option visually too?
                           const nearMeOption = sortSelect.querySelector('option[value="nearme"]');
                           if (nearMeOption) nearMeOption.disabled = true;
                     }
                    await this.sortRows('popularity'); // Re-sort by popularity
                    return; // Exit early
                }
                // Sort by pre-calculated distance stored in data-distance
                this.visibleRows.sort((a, b) => {
                    const distA = parseFloat(a.dataset.distance);
                    const distB = parseFloat(b.dataset.distance);

                    // Handle NaN or infinite values (put them at the end)
                    if (isNaN(distA) || !isFinite(distA)) return 1;
                    if (isNaN(distB) || !isFinite(distB)) return -1;

                    return distA - distB; // Ascending order (nearest first)
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
                // Date-based sorting ('newest', 'oldest')
                this.visibleRows.sort((a, b) => {
                    // Handle 'N/A' dates by treating them as very old/very new depending on sort
                    const dateStrA = a.dataset.date;
                    const dateStrB = b.dataset.date;

                    if (dateStrA === 'N/A' && dateStrB === 'N/A') return 0;
                    if (dateStrA === 'N/A') return sortType === 'newest' ? 1 : -1; // N/A is oldest
                    if (dateStrB === 'N/A') return sortType === 'newest' ? -1 : 1; // N/A is oldest

                    const dateA = new Date(dateStrA);
                    const dateB = new Date(dateStrB);

                    // Add safety checks for invalid date parsing just in case
                    if (isNaN(dateA) && isNaN(dateB)) return 0;
                    if (isNaN(dateA)) return sortType === 'newest' ? 1 : -1;
                    if (isNaN(dateB)) return sortType === 'newest' ? -1 : 1;


                    return sortType === 'newest' ? dateB - dateA : dateA - dateB;
                });
            }


            const tbody = document.querySelector('#data-table tbody');
            // Detach all rows first for performance
            const fragment = document.createDocumentFragment();
            this.visibleRows.forEach(row => fragment.appendChild(row));
            // Clear the tbody completely
            while (tbody.firstChild) {
                 tbody.removeChild(tbody.firstChild);
            }
            // Append the sorted rows back
            tbody.appendChild(fragment);


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
                if (pattern) { // Only filter if pattern is valid
                    filteredRows = filteredRows.filter(row => {
                        // Improve search by checking specific columns if needed, or just textContent
                        const text = row.textContent || row.innerText || '';
                        return pattern.test(text);
                    });
                }
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
            await this.sortRows(this.currentSort); // sortRows now updates the table internally

            // Reset to first page and update display (This might be redundant now)
            // this.currentPage = 1; // sortRows already sets this
            // this.updateTable(); // sortRows already calls this
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

            // Read the CURRENT sort value from the dropdown
            const sortSelect = document.getElementById('sortFilter');
            this.currentSort = sortSelect ? sortSelect.value : 'popularity';

            // Add range filters
            const rangeFilters = {
                 pledged: {
                      min: parseFloat(document.getElementById('fromInput').value),
                      max: parseFloat(document.getElementById('toInput').value)
                 },
                 goal: {
                      min: parseFloat(document.getElementById('goalFromInput').value),
                      max: parseFloat(document.getElementById('goalToInput').value)
                 },
                 raised: {
                      min: parseFloat(document.getElementById('raisedFromInput').value),
                      max: parseFloat(document.getElementById('raisedToInput').value)
                 }
            };
            // Validate range filters to ensure min <= max
            if (rangeFilters.pledged.min > rangeFilters.pledged.max) rangeFilters.pledged.min = rangeFilters.pledged.max;
            if (rangeFilters.goal.min > rangeFilters.goal.max) rangeFilters.goal.min = rangeFilters.goal.max;
            if (rangeFilters.raised.min > rangeFilters.raised.max) rangeFilters.raised.min = rangeFilters.raised.max;

            this.currentFilters.ranges = rangeFilters;


            await this.applyAllFilters();
        }

        initialize() {
            this.setupSearchAndPagination();
            this.setupFilters();
            this.setupRangeSlider();
            this.currentSort = 'popularity';  // Set default sort to popularity

             // Disable 'Near Me' initially if no location
            const sortSelect = document.getElementById('sortFilter');
            const nearMeOption = sortSelect ? sortSelect.querySelector('option[value="nearme"]') : null;
            if (nearMeOption && !window.hasLocation) {
                 nearMeOption.disabled = true;
            }

            this.applyAllFilters(); // This now includes sorting and updating the table
            // this.updateTable(); // No longer needed here, called by applyAllFilters/sortRows
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
            // Define the function globally so inline onclick can access it
            window.handlePageClick = (page) => {
                if (window.tableManager) { // Ensure tableManager is initialized
                    window.tableManager.goToPage(page);
                }
            };
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

            // Country filter - Use data attribute for consistency if available
            const countryCode = row.dataset.countryCode; // Use country code if reliable
            const countryText = row.querySelector('td:nth-child(5)')?.textContent?.trim(); // Fallback to text
            const countryToMatch = countryCode !== 'N/A' ? countryCode : countryText; // Prefer code? Or text? Check consistency with filter options. Assuming filter uses full names from `country_expanded_col`
            if (!filters.countries.includes('All Countries') && !filters.countries.includes(countryText)) { // Match based on text displayed / generated in options
                return false;
            }


            // State filter - Extract state from class name
            const stateCell = row.querySelector('.state_cell');
            let state = 'unknown'; // Default if no cell or class found
            if (stateCell) {
                 // Find class like 'state-successful'
                 const stateClass = Array.from(stateCell.classList).find(cls => cls.startsWith('state-'));
                 if (stateClass) {
                      state = stateClass.substring(6); // e.g., 'successful'
                 }
            }
            // Compare lowercase state name with capitalized filter options (lowercase both)
            if (!filters.states.includes('All States')) {
                 const stateLower = state.toLowerCase();
                 const matchingState = filters.states.find(s => s.toLowerCase() === stateLower);
                 if (!matchingState) return false;
            }


            // Get all other values safely
            const pledged = parseFloat(row.dataset.pledged || '0'); // Default to 0 if missing
            const goal = parseFloat(row.dataset.goal || '0');
            const raised = parseFloat(row.dataset.raised || '0');
            const dateStr = row.dataset.date; // Keep as string for now


            // Rest of filter checks
            // Check range filters (use values from filters.ranges)
            const ranges = filters.ranges || {}; // Ensure ranges object exists
            const pledgedRange = ranges.pledged || { min: -Infinity, max: Infinity };
            const goalRange = ranges.goal || { min: -Infinity, max: Infinity };
            const raisedRange = ranges.raised || { min: -Infinity, max: Infinity };

            if (pledged < pledgedRange.min || pledged > pledgedRange.max) return false;
            if (goal < goalRange.min || goal > goalRange.max) return false;

            // Handle 0% raised correctly
            if (raised === 0 && raisedRange.min > 0) return false;
            if (raised < raisedRange.min || raised > raisedRange.max) return false;


            // Date filter
            if (filters.date !== 'All Time') {
                 if (dateStr === 'N/A') return false; // Don't include N/A dates in specific ranges
                 const date = new Date(dateStr);
                 if (isNaN(date)) return false; // Invalid date format

                const now = new Date();
                let compareDate = new Date();
                // Set time to 00:00:00 for consistent date comparisons
                now.setHours(0,0,0,0);
                compareDate.setHours(0,0,0,0);
                date.setHours(0,0,0,0);


                switch(filters.date) {
                    case 'Last Month': compareDate.setMonth(now.getMonth() - 1); break;
                    case 'Last 6 Months': compareDate.setMonth(now.getMonth() - 6); break;
                    case 'Last Year': compareDate.setFullYear(now.getFullYear() - 1); break;
                    case 'Last 5 Years': compareDate.setFullYear(now.getFullYear() - 5); break;
                    case 'Last 10 Years': compareDate.setFullYear(now.getFullYear() - 10); break;
                    default: return true; // Should not happen with current options
                }

                if (date < compareDate) return false;
            }

            return true; // If all checks pass
        }


        resetFilters() {
            // Reset category selections
            const categoryOptions = document.querySelectorAll('.category-option');
            categoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCategoriesOption = document.querySelector('.category-option[data-value="All Categories"]');
            if (allCategoriesOption) allCategoriesOption.classList.add('selected'); // Add check
            const categoryBtn = document.getElementById('categoryFilterBtn'); // Use ID
            if (categoryBtn) categoryBtn.textContent = 'All Categories'; // Add check

            // Reset country selections
            const countryOptions = document.querySelectorAll('.country-option');
            countryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCountriesOption = document.querySelector('.country-option[data-value="All Countries"]');
            if (allCountriesOption) allCountriesOption.classList.add('selected');
            const countryBtn = document.getElementById('countryFilterBtn'); // Use ID
             if (countryBtn) countryBtn.textContent = 'All Countries';


            // Reset state selections
            const stateOptions = document.querySelectorAll('.state-option');
            stateOptions.forEach(opt => opt.classList.remove('selected'));
            const allStatesOption = document.querySelector('.state-option[data-value="All States"]');
            if (allStatesOption) allStatesOption.classList.add('selected');
            const stateBtn = document.getElementById('stateFilterBtn'); // Use ID
            if (stateBtn) stateBtn.textContent = 'All States';


            // Reset subcategory selections
            const subcategoryOptions = document.querySelectorAll('.subcategory-option');
            subcategoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allSubcategoriesOption = document.querySelector('.subcategory-option[data-value="All Subcategories"]');
             if (allSubcategoriesOption) allSubcategoriesOption.classList.add('selected');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn'); // Use ID
            if (subcategoryBtn) subcategoryBtn.textContent = 'All Subcategories';


            // Reset the stored selections in the Sets (if still used, maybe remove if filters read directly)
            // If applyFilters reads selections directly, these Sets might be redundant
            if (window.selectedCategories) window.selectedCategories.clear(); window.selectedCategories.add('All Categories');
            if (window.selectedCountries) window.selectedCountries.clear(); window.selectedCountries.add('All Countries');
            if (window.selectedStates) window.selectedStates.clear(); window.selectedStates.add('All States');
            if (window.selectedSubcategories) window.selectedSubcategories.clear(); window.selectedSubcategories.add('All Subcategories');

            // Reset range sliders and inputs
             this.resetRangeSliders();


            // Reset Sort dropdown to 'popularity'
            const sortSelect = document.getElementById('sortFilter');
             if (sortSelect) sortSelect.value = 'popularity';


             // Reset Date Filter dropdown to 'All Time'
            const dateSelect = document.getElementById('dateFilter');
            if (dateSelect) dateSelect.value = 'All Time';


            this.searchInput.value = '';
            this.currentSearchTerm = '';
            this.currentFilters = null; // Clear applied filters state
            this.currentSort = 'popularity'; // Reset sort state
            // this.visibleRows = this.allRows; // applyAllFilters will handle this
            this.applyAllFilters(); // Re-apply default filters/sort
        }

         resetRangeSliders() {
             if (this.rangeSliderElements) {
                 const {
                     fromSlider, toSlider, fromInput, toInput,
                     goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                     raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                     fillSlider // Make sure fillSlider is correctly referenced or defined
                 } = this.rangeSliderElements;

                 // Helper to safely get min/max or default
                 const getAttr = (el, attr, defaultVal) => el ? (el[attr] || defaultVal) : defaultVal;

                 // Reset pledged amount range
                 const minPledged = getAttr(fromSlider, 'min', 0);
                 const maxPledged = getAttr(toSlider, 'max', 1000);
                 if (fromSlider) fromSlider.value = minPledged;
                 if (toSlider) toSlider.value = maxPledged;
                 if (fromInput) fromInput.value = minPledged;
                 if (toInput) toInput.value = maxPledged;
                 if (fromSlider && toSlider && fillSlider) fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);


                 // Reset goal amount range
                 const minGoal = getAttr(goalFromSlider, 'min', 0);
                 const maxGoal = getAttr(goalToSlider, 'max', 10000);
                 if (goalFromSlider) goalFromSlider.value = minGoal;
                 if (goalToSlider) goalToSlider.value = maxGoal;
                 if (goalFromInput) goalFromInput.value = minGoal;
                 if (goalToInput) goalToInput.value = maxGoal;
                 if (goalFromSlider && goalToSlider && fillSlider) fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);


                 // Reset percentage raised range
                 const minRaised = getAttr(raisedFromSlider, 'min', 0);
                 const maxRaised = getAttr(raisedToSlider, 'max', 500); // Check if this max corresponds to slider definition
                 if (raisedFromSlider) raisedFromSlider.value = minRaised;
                 if (raisedToSlider) raisedToSlider.value = maxRaised;
                 if (raisedFromInput) raisedFromInput.value = minRaised;
                 if (raisedToInput) raisedToInput.value = maxRaised;
                 if (raisedFromSlider && raisedToSlider && fillSlider) fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
             } else {
                  console.warn("Range slider elements not found for reset.");
             }
         }


        updateTable() {
             const tbody = document.querySelector('#data-table tbody');
             if (!tbody) {
                 console.error("Table body not found for updating.");
                 return;
             }

            // Hide all rows currently in the visibleRows array (which might just have been sorted)
             // This prevents brief display of rows not on the current page
             this.visibleRows.forEach(row => {
                 // Ensure row is still attached to the DOM before trying to hide
                 if (row.parentNode === tbody) {
                      row.style.display = 'none';
                 }
             });


            // Calculate visible range based on current page and page size
            const start = (this.currentPage - 1) * this.pageSize;
            const end = Math.min(start + this.pageSize, this.visibleRows.length);

            // Show only rows for the current page slice
            this.visibleRows.slice(start, end).forEach(row => {
                 // Ensure row is still attached to the DOM before trying to show
                 if (row.parentNode === tbody) {
                     row.style.display = ''; // Show row by resetting display style
                 } else {
                      // If a row isn't in the tbody (e.g., after a sort), it shouldn't be shown
                      console.warn("Attempted to show a row not currently in the table body.");
                 }
            });

            this.updatePagination();
            this.adjustHeight();
        }

        updatePagination() {
            const totalPages = Math.max(1, Math.ceil(this.visibleRows.length / this.pageSize));
            const pageNumbers = this.generatePageNumbers(totalPages);
            const container = document.getElementById('page-numbers');
            if (!container) return; // Safety check

            container.innerHTML = pageNumbers.map(page => {
                if (page === '...') {
                    return '<span class="page-ellipsis">...</span>';
                }
                // Ensure onclick calls the globally defined function
                return `<button class="page-number ${page === this.currentPage ? 'active' : ''}"
                    ${page === this.currentPage ? 'disabled' : ''}
                    onclick="window.handlePageClick(${page})">${page}</button>`;
            }).join('');

            const prevButton = document.getElementById('prev-page');
            const nextButton = document.getElementById('next-page');
            if (prevButton) prevButton.disabled = this.currentPage <= 1;
            if (nextButton) nextButton.disabled = this.currentPage >= totalPages;

        }

        generatePageNumbers(totalPages) {
            let pages = [];
            const C = this.currentPage; // Current Page
            const T = totalPages;      // Total Pages
            const S = 7; // Number of pages to show including ellipsis (e.g., 1 ... 4 5 6 ... 10) - must be odd >= 5

             if (T <= S + 2) { // Show all pages if total is small enough
                 pages = Array.from({length: T}, (_, i) => i + 1);
             } else {
                 const K = Math.floor((S - 3) / 2); // Number of pages around current page (excluding current)

                 if (C <= K + 2) { // Near the beginning
                     pages = [...Array.from({length: K + 3}, (_, i) => i + 1), '...', T];
                 } else if (C >= T - (K + 1)) { // Near the end
                     pages = [1, '...', ...Array.from({length: K + 3}, (_, i) => T - (K + 2) + i)];
                 } else { // In the middle
                     pages = [1, '...', ...Array.from({length: K * 2 + 1}, (_, i) => C - K + i), '...', T];
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
            if (page >= 1 && page <= totalPages && page !== this.currentPage) { // Only update if page changed
                this.currentPage = page;
                this.updateTable();
            }
        }

        adjustHeight() {
             // Debounce or throttle this if it causes performance issues on resize
            requestAnimationFrame(() => {
                const elements = {
                    titleWrapper: document.querySelector('.title-wrapper'),
                    filterWrapper: document.querySelector('.filter-wrapper'),
                    tableWrapper: document.querySelector('.table-wrapper'),
                    tableContainer: document.querySelector('.table-container'),
                    // table: document.querySelector('#data-table'), // Table itself might not be needed
                    controls: document.querySelector('.table-controls'),
                    pagination: document.querySelector('.pagination-controls'),
                    tbody: document.querySelector('#data-table tbody') // Need tbody to estimate row height
                };

                // Check essential elements for height calculation
                if (!elements.titleWrapper || !elements.filterWrapper || !elements.tableWrapper || !elements.tableContainer || !elements.controls || !elements.pagination || !elements.tbody) {
                    // console.warn("One or more elements missing for height adjustment.");
                    // Try to set a default height if possible or just return
                    // Streamlit.setFrameHeight(window.innerHeight); // Fallback maybe?
                    return;
                }


                // Estimate row height from the first visible row if possible
                let rowHeight = 52; // Default fallback
                const firstVisibleRow = elements.tbody.querySelector('tr:not([style*="display: none"])');
                 if (firstVisibleRow) {
                      const style = window.getComputedStyle(firstVisibleRow);
                      // Include margins if they affect layout
                      const marginTop = parseFloat(style.marginTop);
                      const marginBottom = parseFloat(style.marginBottom);
                      rowHeight = firstVisibleRow.offsetHeight + marginTop + marginBottom;
                 } else if (this.visibleRows.length > 0 && this.allRows[0]) {
                      // If no rows are visible *yet* (e.g., page loading), estimate from first overall row
                      // Temporarily show it to measure? Risky. Use default or estimate from first row data.
                      rowHeight = 52; // Stick to default if no visible rows
                 }


                // Count visible rows on the current page
                const start = (this.currentPage - 1) * this.pageSize;
                const end = Math.min(start + this.pageSize, this.visibleRows.length);
                const visibleRowCount = end - start;


                // Constants
                const headerHeight = document.querySelector('#data-table thead')?.offsetHeight || 60; // Measure header or use default
                const controlsHeight = elements.controls.offsetHeight;
                const paginationHeight = elements.pagination.offsetHeight;
                const wrapperPadding = 40; // Combined padding/margin for the main wrapper
                const minTableContainerHeight = 300;  // Minimum height for the scrollable table area

                // Calculate table content height
                // Use actual number of rows displayed, max 10 (pageSize)
                const tableContentHeight = (visibleRowCount * rowHeight) + headerHeight;
                const actualTableContainerHeight = Math.max(tableContentHeight, minTableContainerHeight);

                // Set dimensions for the scrollable container
                 elements.tableContainer.style.height = `${actualTableContainerHeight}px`;
                 // The wrapper height is determined by its content naturally + container height
                 // elements.tableWrapper.style.height = `${actualTableContainerHeight + controlsHeight + paginationHeight}px`; // This might constrain it too much

                // Calculate final component height needed for Streamlit frame
                // Sum heights of non-scrollable parts + the calculated scrollable height + padding
                 const finalHeight =
                     elements.titleWrapper.offsetHeight +
                     elements.filterWrapper.offsetHeight +
                     controlsHeight + // Table controls are outside scrolling area
                     actualTableContainerHeight + // The scrolling container height
                     paginationHeight + // Pagination controls are outside scrolling area
                     wrapperPadding; // Overall padding/margins


                // Update Streamlit frame height if changed significantly
                 const currentFrameHeight = document.documentElement.scrollHeight; // Get current frame height
                 if (!this.lastHeight || Math.abs(this.lastHeight - finalHeight) > 20) { // Increase threshold slightly
                     console.log(`Adjusting height from ${this.lastHeight} to ${finalHeight}`);
                     this.lastHeight = finalHeight;
                     Streamlit.setFrameHeight(finalHeight);
                 } else if (Math.abs(currentFrameHeight - finalHeight) > 20) {
                      // If calculated height differs significantly from actual scroll height, update anyway
                      console.log(`Adjusting height (scroll diff) from ${currentFrameHeight} to ${finalHeight}`);
                      this.lastHeight = finalHeight;
                      Streamlit.setFrameHeight(finalHeight);
                 }
            });
        }


        setupFilters() {
            // Remove global Sets - read directly from elements when applying filters
            // window.selectedCategories = new Set(['All Categories']); // REMOVE
            // window.selectedCountries = new Set(['All Countries']); // REMOVE
            // window.selectedStates = new Set(['All States']); // REMOVE
            // window.selectedSubcategories = new Set(['All Subcategories']); // REMOVE

            // Get button elements by ID more robustly
            const categoryBtn = document.getElementById('categoryFilterBtn');
            const countryBtn = document.getElementById('countryFilterBtn');
            const stateBtn = document.getElementById('stateFilterBtn');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            // Update button text function
            const updateButtonText = (optionsSelector, buttonElement, allValue) => {
                if (!buttonElement) return;

                const selectedOptions = Array.from(document.querySelectorAll(`${optionsSelector}.selected`));
                const selectedValues = selectedOptions.map(opt => opt.dataset.value);

                if (selectedValues.includes(allValue) || selectedValues.length === 0) {
                    buttonElement.textContent = allValue;
                } else {
                     // Sort selected values alphabetically for consistent display
                     selectedValues.sort((a, b) => a.localeCompare(b));
                    if (selectedValues.length > 2) {
                        buttonElement.textContent = `${selectedValues[0]}, ${selectedValues[1]} +${selectedValues.length - 2}`;
                    } else {
                        buttonElement.textContent = selectedValues.join(', ');
                    }
                }
            };


            // Setup multi-select handlers
            const setupMultiSelect = (optionsSelector, allValue, buttonElement) => {
                const options = document.querySelectorAll(optionsSelector);
                 if (options.length === 0) {
                      console.warn(`No options found for selector: ${optionsSelector}`);
                      return; // Don't setup if no options
                 }

                const allOption = document.querySelector(`${optionsSelector}[data-value="${allValue}"]`);

                 // Ensure "All" option is selected by default visually
                 if (allOption) {
                      options.forEach(opt => opt.classList.remove('selected')); // Clear others first
                      allOption.classList.add('selected');
                 } else {
                      console.warn(`"All" option not found for ${optionsSelector}`);
                 }


                options.forEach(option => {
                    option.addEventListener('click', (e) => {
                        const clickedValue = e.target.dataset.value;
                        const isSelected = e.target.classList.contains('selected');

                        if (clickedValue === allValue) {
                            // If "All" is clicked, deselect others and select "All"
                            options.forEach(opt => opt.classList.remove('selected'));
                            if(allOption) allOption.classList.add('selected');
                        } else {
                            // If a specific option is clicked
                            e.target.classList.toggle('selected'); // Toggle its state

                            // If this action resulted in selecting it, deselect "All"
                            if (e.target.classList.contains('selected') && allOption) {
                                allOption.classList.remove('selected');
                            }

                            // Check if any specific options are selected
                            const anySelected = Array.from(options).some(opt => opt.dataset.value !== allValue && opt.classList.contains('selected'));

                            // If no specific options are selected, select "All"
                            if (!anySelected && allOption) {
                                allOption.classList.add('selected');
                            }
                        }

                        updateButtonText(optionsSelector, buttonElement, allValue);
                        this.applyFilters(); // Trigger filter application
                    });
                });

                // Initialize button text on load
                updateButtonText(optionsSelector, buttonElement, allValue);
            };

            // Setup each multi-select
            setupMultiSelect('.category-option', 'All Categories', categoryBtn);
            setupMultiSelect('.country-option', 'All Countries', countryBtn);
            setupMultiSelect('.state-option', 'All States', stateBtn);
            setupMultiSelect('.subcategory-option', 'All Subcategories', subcategoryBtn);


            // Setup other filters (Date and Sort)
            const filterIds = ['dateFilter', 'sortFilter'];
            filterIds.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                     // Use 'change' event for select dropdowns
                    element.addEventListener('change', () => this.applyFilters());
                } else {
                     console.warn(`Filter element with ID "${id}" not found.`);
                }
            });


            // Add reset button handler
            const resetButton = document.getElementById('resetFilters');
            if (resetButton) {
                resetButton.addEventListener('click', () => this.resetFilters());
            } else {
                 console.warn("Reset button not found.");
            }

            // Initialize Range Sliders (ensure this happens after elements are in DOM)
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

            // Check if all elements exist before proceeding
            if (!fromSlider || !toSlider || !fromInput || !toInput ||
                !goalFromSlider || !goalToSlider || !goalFromInput || !goalToInput ||
                !raisedFromSlider || !raisedToSlider || !raisedFromInput || !raisedToInput) {
                console.error("One or more range slider elements are missing. Cannot initialize sliders.");
                this.rangeSliderElements = null; // Indicate sliders are not set up
                return;
            }


            let inputTimeout;

            const fillSlider = (from, to, sliderColor, rangeColor, controlSlider) => {
                 // Ensure sliders exist before accessing properties
                 if (!from || !to || !controlSlider) return;

                 const rangeDistance = parseFloat(controlSlider.max) - parseFloat(controlSlider.min);
                 const fromPosition = parseFloat(from.value) - parseFloat(controlSlider.min);
                 const toPosition = parseFloat(to.value) - parseFloat(controlSlider.min);

                 // Handle potential division by zero or invalid range
                 if (rangeDistance <= 0) {
                      // Set a default background or handle appropriately
                      controlSlider.style.background = sliderColor;
                      return;
                 }


                 // Calculate percentages
                 const fromPercent = (fromPosition / rangeDistance) * 100;
                 const toPercent = (toPosition / rangeDistance) * 100;


                 controlSlider.style.background = `linear-gradient(
                    to right,
                    ${sliderColor} ${fromPercent}%,
                    ${rangeColor} ${fromPercent}%,
                    ${rangeColor} ${toPercent}%,
                    ${sliderColor} ${toPercent}%)`;

            };

             // Store fillSlider in the instance so it's accessible in resetRangeSliders
             this.rangeSliderElements = { fillSlider };


            const debouncedApplyFilters = debounce(() => this.applyFilters(), 500); // Slightly longer debounce for sliders/inputs

            const controlFromSlider = (fromSlider, toSlider, fromInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider); // Use the instance fillSlider
                if (from > to) {
                    fromSlider.value = to;
                    if(fromInput) fromInput.value = to; // Check if input exists
                } else {
                     if(fromInput) fromInput.value = from;
                }
                 debouncedApplyFilters(); // Apply filters after slider changes
            };

            const controlToSlider = (fromSlider, toSlider, toInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider); // Use the instance fillSlider
                if (from <= to) {
                     if(toInput) toInput.value = to; // Check if input exists
                    // Don't need to set toSlider.value = to, it's already set by user action
                } else {
                     if(toInput) toInput.value = from;
                    toSlider.value = from; // Adjust the 'to' slider if it's less than 'from'
                }
                 debouncedApplyFilters(); // Apply filters after slider changes
            };

            const getParsedValue = (fromSlider, toSlider) => {
                 // Provide defaults if sliders don't exist (shouldn't happen with initial check)
                 const fromVal = fromSlider ? parseFloat(fromSlider.value) : 0;
                 const toVal = toSlider ? parseFloat(toSlider.value) : 0;
                 return [fromVal, toVal];
            };

            const validateAndUpdateRangeInput = (inputElement, isMin = true) => {
                 clearTimeout(inputTimeout); // Clear previous timeout

                 inputTimeout = setTimeout(() => {
                      if (!inputElement) return;

                      let value = parseFloat(inputElement.value);
                      const minAllowed = parseFloat(inputElement.min);
                      const maxAllowed = parseFloat(inputElement.max);

                      // Determine corresponding sliders based on input ID pattern
                      let fromSliderElement, toSliderElement;
                      if (inputElement.id.includes('goal')) {
                           fromSliderElement = goalFromSlider;
                           toSliderElement = goalToSlider;
                      } else if (inputElement.id.includes('raised')) {
                           fromSliderElement = raisedFromSlider;
                           toSliderElement = raisedToSlider;
                      } else { // Default to pledged
                           fromSliderElement = fromSlider;
                           toSliderElement = toSlider;
                      }

                      if (!fromSliderElement || !toSliderElement) return; // Need both sliders


                      if (isNaN(value)) {
                           value = isMin ? minAllowed : maxAllowed; // Default to min/max if input is invalid
                      }

                      // Clamp value within min/max allowed for the input itself
                      value = Math.max(minAllowed, Math.min(maxAllowed, value));


                      if (isMin) {
                           const maxValue = parseFloat(toSliderElement.value);
                           // Ensure 'from' value isn't greater than the 'to' slider's current value
                           value = Math.min(value, maxValue);
                           fromSliderElement.value = value; // Update the 'from' slider
                      } else { // isMax
                           const minValue = parseFloat(fromSliderElement.value);
                           // Ensure 'to' value isn't less than the 'from' slider's current value
                           value = Math.max(value, minValue);
                           toSliderElement.value = value; // Update the 'to' slider
                      }

                      inputElement.value = value; // Update input field with validated/adjusted value
                      fillSlider(fromSliderElement, toSliderElement, '#C6C6C6', '#5932EA', toSliderElement); // Update slider background
                      debouncedApplyFilters(); // Apply filters after validation
                 }, 750); // Delay before validating input (e.g., 750ms)
            };


            // --- Event Listeners ---

             // Slider Input Listeners
             fromSlider.addEventListener('input', () => controlFromSlider(fromSlider, toSlider, fromInput));
             toSlider.addEventListener('input', () => controlToSlider(fromSlider, toSlider, toInput));
             goalFromSlider.addEventListener('input', () => controlFromSlider(goalFromSlider, goalToSlider, goalFromInput));
             goalToSlider.addEventListener('input', () => controlToSlider(goalFromSlider, goalToSlider, goalToInput));
             raisedFromSlider.addEventListener('input', () => controlFromSlider(raisedFromSlider, raisedToSlider, raisedFromInput));
             raisedToSlider.addEventListener('input', () => controlToSlider(raisedFromSlider, raisedToSlider, raisedToInput));

             // Input Field Listeners (using 'input' for responsiveness, validated on timeout)
             fromInput.addEventListener('input', () => validateAndUpdateRangeInput(fromInput, true));
             toInput.addEventListener('input', () => validateAndUpdateRangeInput(toInput, false));
             goalFromInput.addEventListener('input', () => validateAndUpdateRangeInput(goalFromInput, true));
             goalToInput.addEventListener('input', () => validateAndUpdateRangeInput(goalToInput, false));
             raisedFromInput.addEventListener('input', () => validateAndUpdateRangeInput(raisedFromInput, true));
             raisedToInput.addEventListener('input', () => validateAndUpdateRangeInput(raisedToInput, false));


            // Store references for reset function - Ensure fillSlider is stored correctly
            this.rangeSliderElements = {
                fromSlider, toSlider, fromInput, toInput,
                goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                fillSlider // Store the function itself
            };

            // Initial setup of slider backgrounds
            fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
            fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);
            fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
        }
    }

    // --- Initialization ---
    function onRender(event) {
        // Ensure initialization runs only once
        if (!window.rendered) {
             console.log("Initializing TableManager...");
            window.tableManager = new TableManager(); // Creates and initializes
            window.rendered = true;

            // Add resize observer for dynamic height adjustment
            const resizeObserver = new ResizeObserver(() => {
                if (window.tableManager) {
                    window.tableManager.adjustHeight();
                }
            });
            // Observe the main app container or a suitable wrapper
            const appContainer = document.querySelector('[data-testid="stAppViewContainer"]'); // Or find a more specific wrapper if needed
            if (appContainer) {
                 resizeObserver.observe(appContainer);
            } else {
                 console.error("App container not found for ResizeObserver.");
            }
        } else {
             console.log("TableManager already initialized.");
             // Potentially call adjustHeight here if re-renders might change layout
             if (window.tableManager) {
                  window.tableManager.adjustHeight();
             }
        }
    }

    // Streamlit event listeners
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();

"""

# Create and use the component
table_component = generate_component('searchable_table', template=css + template, script=script)
table_component(key="kickstarter_table") # Add a key for stability