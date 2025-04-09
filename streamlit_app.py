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
def load_data_from_parquet_chunks() -> tuple[pl.LazyFrame | None, str | None]: # Return LazyFrame and temp file path
    """
    Scan data from compressed parquet chunks lazily.
    Combines chunks first, then scans the resulting file.
    Returns a tuple: (LazyFrame, temp_file_path_to_delete_later)
    """
    chunk_files = glob.glob("parquet_gz_chunks/*.part")

    if not chunk_files:
        st.error("No parquet chunks found in parquet_gz_chunks folder. Please run database_download.py first.")
        return None, None # Indicate error

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Found {len(chunk_files)} chunk files. Combining chunks...")

    combined_filename = None
    decompressed_file_obj = None # To hold the file object
    decompressed_filename = None # To hold the file path
    lf = None

    try:
        # Create a temporary file for the combined chunks (deleted automatically on close)
        with tempfile.NamedTemporaryFile(delete=True, suffix='.parquet.gz') as combined_file:
            combined_filename = combined_file.name # Get path for potential logging
            chunk_files = sorted(chunk_files)
            for i, chunk_file in enumerate(chunk_files):
                try:
                    with open(chunk_file, 'rb') as f:
                        combined_file.write(f.read())
                    progress_bar.progress((i + 1) / (2 * len(chunk_files)))
                    status_text.text(f"Combined chunk {i+1}/{len(chunk_files)}")
                except Exception as e:
                    st.warning(f"Error reading chunk {chunk_file}: {str(e)}")

            combined_file.flush() # Ensure all data is written before gzip reads it

            status_text.text("Decompressing combined file...")
            # Create the decompressed file - IMPORTANT: delete=False
            decompressed_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
            decompressed_filename = decompressed_file_obj.name # Store the path

            with gzip.open(combined_filename, 'rb') as gz_file:
                 # Write decompressed data to the persistent temp file
                decompressed_file_obj.write(gz_file.read())
            decompressed_file_obj.close() # Close the file handle, but file persists


        # Scan the decompressed parquet file using Polars Lazily
        status_text.text("Scanning pre-processed parquet data...")
        progress_bar.progress(0.75)

        lf = pl.scan_parquet(decompressed_filename)

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # Return the LazyFrame and the path to the temp file that needs deletion later
        return lf, decompressed_filename

    except Exception as e:
        st.error(f"Error processing parquet file for scanning: {str(e)}")
         # Clean up the decompressed file if it exists and an error occurred
        if decompressed_filename and os.path.exists(decompressed_filename):
            try:
                os.unlink(decompressed_filename)
            except OSError as unlink_error:
                st.warning(f"Error cleaning up temporary decompressed file on error: {unlink_error}")
        return None, None # Indicate error
    # No finally block needed here for decompressed_filename, as we return its path


# Load pre-processed data lazily
lf, temp_parquet_path = load_data_from_parquet_chunks()

# Exit if loading failed
if lf is None or temp_parquet_path is None:
     st.error("Failed to load or scan data. Cannot proceed.")
     st.stop() # Stop execution if lf is None

# --- Check if LazyFrame is potentially empty ---
try:
    # Apply head(0) lazily, then collect the empty frame with the correct schema
    schema_check = lf.head(0).collect()
    if schema_check.is_empty() and schema_check.width == 0 :
         st.error("Loaded data appears empty or schema is invalid. Please check logs and ensure database_download.py ran successfully.")
         # Clean up the temp file before stopping
         if temp_parquet_path and os.path.exists(temp_parquet_path):
              try:
                  os.unlink(temp_parquet_path)
              except OSError as e:
                  st.warning(f"Error cleaning up temporary file on empty schema check: {e}")
         st.stop()
    print("LazyFrame Schema:", lf.schema)
except Exception as e:
     st.error(f"Error during initial data check: {e}. Cannot proceed.")
     # Clean up the temp file before stopping
     if temp_parquet_path and os.path.exists(temp_parquet_path):
          try:
              os.unlink(temp_parquet_path)
          except OSError as unlink_error:
              st.warning(f"Error cleaning up temporary file on data check error: {unlink_error}")
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
        # 'subcategories': ['All Subcategories'], # Remove static list
        'countries': ['All Countries'],
        'states': ['All States'],
        'date_ranges': [
            'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
            'Last 5 Years', 'Last 10 Years'
        ]
    }
    # Add a structure to hold the category -> subcategory mapping
    category_subcategory_map = {'All Categories': ['All Subcategories']}

    try:
        # Get unique categories first
        if 'Category' in _lf.schema:
            categories_unique = _lf.select(pl.col('Category')).unique().collect()['Category']
            valid_categories = sorted(categories_unique.filter(categories_unique.is_not_null() & (categories_unique != "N/A")).to_list())
            options['categories'] += valid_categories
            # Initialize map keys
            for cat in valid_categories:
                category_subcategory_map[cat] = [] # Start with empty list

        # Get unique Category-Subcategory pairs
        if 'Category' in _lf.schema and 'Subcategory' in _lf.schema:
             cat_subcat_pairs = _lf.select(['Category', 'Subcategory']).unique().drop_nulls().collect()

             # Populate the map
             all_subcategories_set = set() # To collect all unique subcats for 'All Categories'
             for row in cat_subcat_pairs.iter_rows(named=True):
                  category = row['Category']
                  subcategory = row['Subcategory']
                  if category and subcategory and category != "N/A" and subcategory != "N/A":
                       if category in category_subcategory_map:
                           category_subcategory_map[category].append(subcategory)
                       all_subcategories_set.add(subcategory)

             # Add sorted unique subcategories to 'All Categories'
             category_subcategory_map['All Categories'] += sorted(list(all_subcategories_set))

             # Sort subcategories within each category
             for cat in category_subcategory_map:
                 # Keep 'All Subcategories' first if present, sort the rest
                 subcats = category_subcategory_map[cat]
                 if 'All Subcategories' in subcats:
                     prefix = ['All Subcategories']
                     rest = sorted([s for s in subcats if s != 'All Subcategories'])
                     category_subcategory_map[cat] = prefix + rest
                 else:
                     category_subcategory_map[cat] = sorted(subcats)

        # Handle Subcategories if Category doesn't exist (fallback)
        elif 'Subcategory' in _lf.schema and 'Category' not in _lf.schema:
             subcategories_unique = _lf.select(pl.col('Subcategory')).unique().collect()['Subcategory']
             all_subcats = sorted(subcategories_unique.filter(subcategories_unique.is_not_null() & (subcategories_unique != "N/A")).to_list())
             category_subcategory_map['All Categories'] += all_subcats # Add to default

        # Ensure 'All Subcategories' exists if map is otherwise empty for it
        if not category_subcategory_map['All Categories']:
             category_subcategory_map['All Categories'] = ['All Subcategories']


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
         options = {k: v[:1] for k, v in options.items() if k != 'subcategories'} # Keep only "All..." options
         options['date_ranges'] = [ # Restore date ranges
            'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
            'Last 5 Years', 'Last 10 Years'
         ]
         category_subcategory_map = {'All Categories': ['All Subcategories']} # Reset map on error

    # Return both the options dict and the mapping
    return options, category_subcategory_map

# Calculate filter options using the LazyFrame
# Now unpack two return values
filter_options, category_subcategory_map = get_filter_options(lf)

# Calculate Min/Max values lazily
min_pledged, max_pledged = 0, 1000
min_goal, max_goal = 0, 10000
min_raised, max_raised = 0, 500 # Default raised range

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
        ]).collect() # This collect is fine, operates on aggregations

        min_pledged = int(min_max_vals['min_pledged'][0]) if min_max_vals['min_pledged'][0] is not None else 0
        max_pledged = int(min_max_vals['max_pledged'][0]) if min_max_vals['max_pledged'][0] is not None else 1000
        min_goal = int(min_max_vals['min_goal'][0]) if min_max_vals['min_goal'][0] is not None else 0
        max_goal = int(min_max_vals['max_goal'][0]) if min_max_vals['max_goal'][0] is not None else 10000
        min_raised = int(min_max_vals['min_raised'][0]) if min_max_vals['min_raised'][0] is not None else 0
        max_raised_calc_val = min_max_vals['max_raised_calc'][0]
        # Cap max_raised for the slider range
        max_raised = int(max_raised_calc_val) 
        print("Min/max ranges calculated.")
    except Exception as e:
        st.error(f"Error calculating min/max filter ranges: {e}. Using defaults.")
else:
    st.error("Missing columns required for min/max filter ranges in schema. Using defaults.")


# --- Main Collect and Cleanup ---
df_collected = None
try:
    print("Collecting final DataFrame for display...")
    start_collect_time = time.time()
    df_collected = lf.collect(streaming=True)
    collect_duration = time.time() - start_collect_time
    loaded = st.success(f"Loaded {len(df_collected)} projects successfully!")
    time.sleep(1.5)
    loaded.empty()
except Exception as e:
    st.error(f"Error collecting final DataFrame: {e}")
    df_collected = pl.DataFrame()
finally:
    # --- IMPORTANT: Clean up the temporary file AFTER collect attempt ---
    if temp_parquet_path and os.path.exists(temp_parquet_path):
        print(f"Cleaning up temporary file: {temp_parquet_path}")
        try:
            os.unlink(temp_parquet_path)
        except OSError as unlink_error:
            st.warning(f"Error cleaning up temporary file: {unlink_error}")

# Check collection result
if df_collected.is_empty():
     st.error("Data collection resulted in an empty DataFrame or failed. Cannot display table.")
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

template = f"""
<script>
    // RE-ADD user location to JavaScript
    window.userLocation = {json.dumps(user_location) if user_location else 'null'};
    window.hasLocation = {json.dumps(bool(user_location))};
    // ADD category-subcategory mapping
    window.categorySubcategoryMap = {json.dumps(category_subcategory_map)};
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
                <div class="multi-select-content" id="subcategoryOptionsContainer">
                    {' '.join(f'<div class="subcategory-option" data-value="{opt}">{opt}</div>' for opt in category_subcategory_map.get('All Categories', ['All Subcategories']))}
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
                {'<option value="nearme">Near Me</option>' if user_location else ''}
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
# RE-ADD distance-related logic
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
            this.visibleRows = this.allRows; // Initially, all rows are visible
            this.currentPage = 1;
            this.pageSize = 10;
            this.currentSearchTerm = '';
            this.currentFilters = null;
            this.currentSort = 'popularity';
            this.userLocation = window.userLocation;
            this.distanceCache = new DistanceCache();
            this.initialize();
            // Removed resetFilters() call here, initialize calls applyAllFilters which handles initial state
        }

        // sortRows - RE-ADD 'nearme' case
        async sortRows(sortType) {
            // --- Sorting Logic Start ---
            // (This part remains the same - sorts this.visibleRows in place)
            if (sortType === 'popularity') {
                // Sort by popularity score
                this.visibleRows.sort((a, b) => {
                    const scoreA = parseFloat(a.dataset.popularity);
                    const scoreB = parseFloat(b.dataset.popularity);
                    if (isNaN(scoreA)) return 1;
                    if (isNaN(scoreB)) return -1;
                    return scoreB - scoreA; // Descending
                });
            } else if (sortType === 'nearme') { // RE-ADD case
                if (!this.userLocation) {
                    console.warn("Attempted to sort by 'Near Me' without location. Falling back to popularity.");
                    const sortSelect = document.getElementById('sortFilter');
                     if (sortSelect && sortSelect.value === 'nearme') {
                           sortSelect.value = 'popularity';
                           this.currentSort = 'popularity';
                     }
                    await this.sortRows('popularity'); // Re-sort by popularity
                    return; // Exit early
                }
                this.visibleRows.sort((a, b) => {
                    const distA = parseFloat(a.dataset.distance);
                    const distB = parseFloat(b.dataset.distance);
                    if (isNaN(distA) || !isFinite(distA)) return 1;
                    if (isNaN(distB) || !isFinite(distB)) return -1;
                    return distA - distB; // Ascending
                });
            } else if (sortType === 'enddate') {
                this.visibleRows.sort((a, b) => {
                    const deadlineA = new Date(a.dataset.deadline);
                    const deadlineB = new Date(b.dataset.deadline);
                    return deadlineB - deadlineA; // Descending (newest end date first)
                });
            } else if (sortType === 'mostfunded') {
                this.visibleRows.sort((a, b) => {
                    const pledgedA = parseFloat(a.dataset.pledged);
                    const pledgedB = parseFloat(b.dataset.pledged);
                    return pledgedB - pledgedA;  // Descending
                });
            } else if (sortType === 'mostbacked') {
                this.visibleRows.sort((a, b) => {
                    const backersA = parseInt(a.dataset.backers);
                    const backersB = parseInt(b.dataset.backers);
                    return backersB - backersA;  // Descending
                });
            } else { // Date-based sorting (newest/oldest)
                this.visibleRows.sort((a, b) => {
                    const dateA = new Date(a.dataset.date);
                    const dateB = new Date(b.dataset.date);
                    return sortType === 'newest' ? dateB - dateA : dateA - dateB;
                });
            }
            // --- Sorting Logic End ---


            const tbody = document.querySelector('#data-table tbody');
            if (!tbody) return; // Guard clause

            // --- Modification Start: Clear and Rebuild tbody ---
            // Clear the entire table body first to remove any stale rows
            tbody.innerHTML = '';
            // Now append only the currently visible (and sorted) rows from the JS array
            this.visibleRows.forEach(row => tbody.appendChild(row));
            // --- Modification End ---

            // Update current page and pagination
            this.currentPage = 1; // Reset to first page after sorting
            this.updateTable();   // Update display based on new tbody content and page 1
        }

        // applyAllFilters with async sorting
        async applyAllFilters() {
            // Start with all rows from the original full set
            let filteredRows = this.allRows;

            // Apply search filter first
            if (this.currentSearchTerm) {
                const pattern = createRegexPattern(this.currentSearchTerm);
                if (pattern) {
                    filteredRows = filteredRows.filter(row => {
                        const text = row.textContent || row.innerText;
                        return pattern.test(text);
                    });
                }
            }

            // Apply dropdown/range filters
            if (this.currentFilters) {
                filteredRows = filteredRows.filter(row => {
                    return this.matchesFilters(row, this.currentFilters);
                });
            }

            // Update the visibleRows array with the filtered result
            this.visibleRows = filteredRows;

            // Apply the current sort to the filtered rows
            // This will also rebuild the tbody with only the filtered/sorted rows
            await this.sortRows(this.currentSort);

            // Reset to first page (already done in sortRows, but safe to keep here)
            // this.currentPage = 1;
            // Update display (already done in sortRows)
            // this.updateTable();
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
                categories: selectedCategories.length > 0 ? selectedCategories : ['All Categories'], // Ensure default if empty
                subcategories: selectedSubcategories.length > 0 ? selectedSubcategories : ['All Subcategories'], // Ensure default
                countries: selectedCountries.length > 0 ? selectedCountries : ['All Countries'], // Ensure default
                states: selectedStates.length > 0 ? selectedStates : ['All States'], // Ensure default
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
            this.currentFilters.ranges = rangeFilters;

            // Apply all filters and sorting
            await this.applyAllFilters();
        }

        initialize() {
            this.setupSearchAndPagination();
            this.setupFilters(); // This will now handle the hierarchical setup
            this.setupRangeSlider();
            this.currentSort = 'popularity';  // Set default sort
            // Initial population of subcategories based on default 'All Categories'
            this.updateSubcategoryOptions();
             // Apply initial state (default filters, default sort)
            this.applyAllFilters(); // This calls sortRows which calls updateTable
            // this.updateTable(); // No longer needed here, done via applyAllFilters -> sortRows
        }

        setupSearchAndPagination() {
            // Setup search
            const debouncedSearch = debounce(async (searchTerm) => { // Make async
                this.currentSearchTerm = searchTerm;
                await this.applyAllFilters(); // Apply search and other filters
            }, 300);

            this.searchInput.addEventListener('input', (e) => {
                debouncedSearch(e.target.value.trim().toLowerCase());
            });

            // Setup pagination controls
            document.getElementById('prev-page').addEventListener('click', () => this.previousPage());
            document.getElementById('next-page').addEventListener('click', () => this.nextPage());
            // Make handlePageClick global or attach differently if needed in dynamic HTML
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
            // Read from data attribute for consistency if available, otherwise fallback to cell text
            const countryCode = row.dataset.countryCode; // Assume 'Country Code' is reliable
            // We need the country *name* for filtering against the dropdown options
            const countryName = row.querySelector('td:nth-child(5)')?.textContent.trim(); // Get displayed country name
             if (!filters.countries.includes('All Countries') && countryName && !filters.countries.includes(countryName)) {
                 return false;
             }


            // State filter - Extract state from class name
            const stateCell = row.querySelector('.state_cell');
            let state = 'unknown'; // Default if no state cell found or class missing
            if (stateCell) {
                for (const cls of stateCell.classList) {
                    if (cls.startsWith('state-')) {
                        state = cls.substring(6); // e.g., 'successful'
                        break;
                    }
                }
            }
            // Compare against the capitalized state names used in the filter options
            if (!filters.states.includes('All States')) {
                const stateLower = state.toLowerCase();
                // Find a match ignoring case (filters.states contains capitalized versions)
                const matchingState = filters.states.find(s => s.toLowerCase() === stateLower);
                if (!matchingState) return false;
            }


            // Get numeric/date values
            const pledged = parseFloat(row.dataset.pledged);
            const goal = parseFloat(row.dataset.goal);
            const raised = parseFloat(row.dataset.raised); // This is percentage
            const date = new Date(row.dataset.date);

            // Check pledged range
             const minPledged = filters.ranges?.pledged?.min ?? 0;
             const maxPledged = filters.ranges?.pledged?.max ?? Infinity;
             if (isNaN(pledged) || pledged < minPledged || pledged > maxPledged) return false;

             // Check goal range
             const minGoal = filters.ranges?.goal?.min ?? 0;
             const maxGoal = filters.ranges?.goal?.max ?? Infinity;
             if (isNaN(goal) || goal < minGoal || goal > maxGoal) return false;

            // Check raised percentage range
             const minRaised = filters.ranges?.raised?.min ?? 0;
             const maxRaised = filters.ranges?.raised?.max ?? Infinity;
             // Handle NaN for raised percentage - treat as not matching any specific range unless min is 0
             if (isNaN(raised)) {
                 if (minRaised > 0) return false; // Cannot match if min > 0 and value is NaN
             } else {
                 // Handle the case where raised is exactly 0%
                 if (raised === 0 && minRaised > 0) return false;
                 if (raised < minRaised || raised > maxRaised) return false;
             }

            // Date filter
            if (filters.date !== 'All Time') {
                const now = new Date();
                let compareDate = new Date();
                compareDate.setHours(0, 0, 0, 0); // Start of day for comparison

                switch(filters.date) {
                    case 'Last Month': compareDate.setMonth(now.getMonth() - 1); break;
                    case 'Last 6 Months': compareDate.setMonth(now.getMonth() - 6); break;
                    case 'Last Year': compareDate.setFullYear(now.getFullYear() - 1); break;
                    case 'Last 5 Years': compareDate.setFullYear(now.getFullYear() - 5); break;
                    case 'Last 10 Years': compareDate.setFullYear(now.getFullYear() - 10); break;
                }

                 // Check if the row's date is valid and before the comparison date
                 if (isNaN(date.getTime()) || date < compareDate) return false;
            }

            return true; // Row matches all active filters
        }

        async resetFilters() { // Make async
            // Reset category selections
            const categoryOptions = document.querySelectorAll('.category-option');
            categoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCategoriesOption = document.querySelector('.category-option[data-value="All Categories"]');
            if (allCategoriesOption) allCategoriesOption.classList.add('selected');
            window.selectedCategories = new Set(['All Categories']); // Reset JS Set
            const categoryBtn = document.getElementById('categoryFilterBtn');
            this.updateButtonText(window.selectedCategories, categoryBtn, 'All Categories');


            // Reset country selections
            const countryOptions = document.querySelectorAll('.country-option');
            countryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCountriesOption = document.querySelector('.country-option[data-value="All Countries"]');
             if (allCountriesOption) allCountriesOption.classList.add('selected');
            window.selectedCountries = new Set(['All Countries']); // Reset JS Set
            const countryBtn = document.getElementById('countryFilterBtn');
             this.updateButtonText(window.selectedCountries, countryBtn, 'All Countries');

            // Reset state selections
            const stateOptions = document.querySelectorAll('.state-option');
            stateOptions.forEach(opt => opt.classList.remove('selected'));
            const allStatesOption = document.querySelector('.state-option[data-value="All States"]');
             if (allStatesOption) allStatesOption.classList.add('selected');
            window.selectedStates = new Set(['All States']); // Reset JS Set
            const stateBtn = document.getElementById('stateFilterBtn');
             this.updateButtonText(window.selectedStates, stateBtn, 'All States');


            // Reset subcategory selections (will be repopulated by updateSubcategoryOptions)
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');
             window.selectedSubcategories = new Set(['All Subcategories']); // Reset JS Set
            // Update subcategory options based on the reset category ('All Categories')
             this.updateSubcategoryOptions(); // This now resets the UI and button text internally
             // No need to call updateButtonText for subcategory here, updateSubcategoryOptions handles it


            // Reset ranges
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

            // Reset search
            this.searchInput.value = '';
            this.currentSearchTerm = '';

            // Reset date dropdown
            document.getElementById('dateFilter').value = 'All Time';

            // Reset sort dropdown
            document.getElementById('sortFilter').value = 'popularity';
            this.currentSort = 'popularity'; // Ensure internal state matches UI

            // Reset internal filter object
            this.currentFilters = null; // Will be rebuilt by applyFilters

            // Re-apply all filters and sorting with the reset state
            // This will set visibleRows correctly based on allRows and default filters/sort
            await this.applyAllFilters();
        }


        updateTable() {
             // The tbody should already contain exactly the rows in this.visibleRows,
             // thanks to the clear/rebuild logic in sortRows.
             // We just need to apply pagination display styles.
             const tbody = document.querySelector('#data-table tbody');
             if (!tbody) return;

             // Get rows currently in the DOM (which should match this.visibleRows)
             const rowsInTbody = Array.from(tbody.children);

             // Safety check (optional)
             // if (rowsInTbody.length !== this.visibleRows.length) {
             //     console.warn("Mismatch between visibleRows array and tbody children count. A full refresh might be needed if issues persist.");
             //     // Could potentially force a full re-render here by calling applyAllFilters again
             // }

             const start = (this.currentPage - 1) * this.pageSize;
             const end = start + this.pageSize;

             // Show/hide rows based on current page
             rowsInTbody.forEach((row, index) => {
                 // Check if the index is within the current page range
                 row.style.display = (index >= start && index < end) ? '' : 'none';
             });

             this.updatePagination(); // Update page numbers/buttons
             this.adjustHeight();     // Adjust component height
        }


        updatePagination() {
            // Calculate total pages based on the *currently visible* rows count
            const totalPages = Math.max(1, Math.ceil(this.visibleRows.length / this.pageSize));
            const pageNumbers = this.generatePageNumbers(totalPages);
            const container = document.getElementById('page-numbers');
            if (!container) return;

            container.innerHTML = pageNumbers.map(page => {
                if (page === '...') {
                    return '<span class="page-ellipsis">...</span>';
                }
                // Ensure page numbers are treated as numbers for comparison
                const pageNum = Number(page);
                return `<button class="page-number ${pageNum === this.currentPage ? 'active' : ''}"
                    ${pageNum === this.currentPage ? 'disabled' : ''}
                    onclick="handlePageClick(${pageNum})">${pageNum}</button>`;
            }).join('');

            // Disable prev/next buttons appropriately
             const prevButton = document.getElementById('prev-page');
             const nextButton = document.getElementById('next-page');
             if (prevButton) prevButton.disabled = this.currentPage <= 1;
             if (nextButton) nextButton.disabled = this.currentPage >= totalPages;
        }

        generatePageNumbers(totalPages) {
            let pages = [];
            const currentPage = this.currentPage; // Use current page
            const maxPagesToShow = 7; // Max number of direct page links (excluding ellipsis)

            if (totalPages <= maxPagesToShow + 2) { // Show all if not many pages
                pages = Array.from({length: totalPages}, (_, i) => i + 1);
            } else {
                // Determine start and end range around current page
                let startPage, endPage;
                 if (currentPage <= Math.ceil(maxPagesToShow / 2)) {
                     startPage = 1;
                     endPage = maxPagesToShow;
                     pages = [...Array.from({length: endPage}, (_, i) => i + 1), '...', totalPages];
                 } else if (currentPage + Math.floor(maxPagesToShow / 2) >= totalPages) {
                     startPage = totalPages - maxPagesToShow + 1;
                     endPage = totalPages;
                     pages = [1, '...', ...Array.from({length: maxPagesToShow}, (_, i) => startPage + i)];
                 } else {
                     startPage = currentPage - Math.floor(maxPagesToShow / 2);
                     endPage = currentPage + Math.floor(maxPagesToShow / 2);
                      // Adjust if maxPagesToShow is even
                     if (maxPagesToShow % 2 === 0) {
                          startPage = currentPage - (maxPagesToShow / 2) + 1;
                          endPage = currentPage + (maxPagesToShow / 2);
                     } else {
                          startPage = currentPage - Math.floor(maxPagesToShow / 2);
                          endPage = currentPage + Math.floor(maxPagesToShow / 2);
                     }

                     pages = [1, '...', ...Array.from({length: endPage - startPage + 1}, (_, i) => startPage + i), '...', totalPages];
                 }
            }
            return pages;
        }


        previousPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.updateTable(); // Just update display for the new page
            }
        }

        nextPage() {
            const totalPages = Math.ceil(this.visibleRows.length / this.pageSize);
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.updateTable(); // Just update display for the new page
            }
        }

        goToPage(page) {
            const totalPages = Math.ceil(this.visibleRows.length / this.pageSize);
            // Ensure page is a number and within valid range
             const pageNum = Number(page);
            if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= totalPages) {
                this.currentPage = pageNum;
                this.updateTable(); // Just update display for the new page
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

                // Ensure all crucial elements are found
                if (!elements.titleWrapper || !elements.filterWrapper || !elements.tableWrapper || !elements.tableContainer || !elements.table || !elements.controls || !elements.pagination) {
                    console.error("One or more elements needed for height adjustment not found.");
                    // Optional: Attempt to set a default height or skip adjustment
                    // Streamlit.setFrameHeight(800); // Example fallback
                    return;
                }


                // Get rows currently displayed on the page (not just visible in the array)
                 const rowsInTbody = Array.from(elements.tableContainer.querySelectorAll('tbody tr'));
                 const displayedRowCount = rowsInTbody.filter(row => row.style.display !== 'none').length;


                // Constants
                 const rowHeightEstimate = 52; // Approximate height per row (adjust if needed)
                 const headerHeight = elements.table.querySelector('thead')?.offsetHeight ?? 60; // Actual or estimated header height
                 const controlsHeight = elements.controls.offsetHeight;
                 const paginationHeight = elements.pagination.offsetHeight;
                 const basePadding = 40; // Padding around components
                 const minTableContentHeight = 200; // Minimum height for table content area

                // Calculate desired height for table content based on displayed rows
                 const tableContentHeight = Math.max(minTableContentHeight, displayedRowCount * rowHeightEstimate + headerHeight);

                // Set dynamic heights
                 elements.tableContainer.style.height = `${tableContentHeight}px`;
                 // Calculate wrapper height based on dynamic table container + controls + pagination
                 const tableWrapperHeight = tableContentHeight + controlsHeight + paginationHeight;
                 elements.tableWrapper.style.height = `${tableWrapperHeight}px`;

                // Calculate final component height including title and filters
                const finalHeight =
                    elements.titleWrapper.offsetHeight +
                    elements.filterWrapper.offsetHeight +
                    tableWrapperHeight + // Use calculated wrapper height
                    basePadding;


                // Update Streamlit frame height if changed significantly
                if (!this.lastHeight || Math.abs(this.lastHeight - finalHeight) > 10) {
                    this.lastHeight = finalHeight;
                     // Add a minimum height safeguard
                     const safeHeight = Math.max(600, finalHeight); // Ensure minimum height of 600px
                    Streamlit.setFrameHeight(safeHeight);
                }
            });
        }


        setupFilters() {
            // Initialize global Sets to track selections
            window.selectedCategories = new Set(['All Categories']);
            window.selectedCountries = new Set(['All Countries']);
            window.selectedStates = new Set(['All States']);
            window.selectedSubcategories = new Set(['All Subcategories']); // Initialize

            // Get button elements by ID
            const categoryBtn = document.getElementById('categoryFilterBtn');
            const countryBtn = document.getElementById('countryFilterBtn');
            const stateBtn = document.getElementById('stateFilterBtn');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            // Helper to update button text (generalized)
            this.updateButtonText = (selectedItems, buttonElement, allValueLabel) => {
                 if (!buttonElement) return;

                 const selectedArray = Array.from(selectedItems);
                 // Check if 'All' is selected or if the set is empty
                 if (selectedItems.has(allValueLabel) || selectedArray.length === 0) {
                     buttonElement.textContent = allValueLabel;
                 }
                 else {
                     // Filter out the 'All' option if other items are selected (shouldn't be needed if logic is correct, but safe)
                     const displayItems = selectedArray.filter(item => item !== allValueLabel);
                     const sortedArray = displayItems.sort((a, b) => a.localeCompare(b));

                     if (sortedArray.length === 0) { // Fallback if only 'All' was present but somehow missed above
                          buttonElement.textContent = allValueLabel;
                     } else if (sortedArray.length > 2) {
                          buttonElement.textContent = `${sortedArray[0]}, ${sortedArray[1]} +${sortedArray.length - 2}`;
                     } else {
                          buttonElement.textContent = sortedArray.join(', ');
                     }
                 }
            };


            // MODIFIED Setup multi-select handlers
            // Added triggerSubcategoryUpdate flag
             this.setupMultiSelect = (optionsQuerySelector, selectedSet, allValue, buttonElement, triggerSubcategoryUpdate = false) => {
                 const contentDiv = buttonElement?.nextElementSibling;
                 if (!contentDiv) return;

                 // Use event delegation on the content div
                 contentDiv.addEventListener('click', async (e) => { // Make async
                     const targetOption = e.target.closest('[data-value]'); // Find the option element clicked
                     if (!targetOption) return; // Clicked outside an option

                     const clickedValue = targetOption.dataset.value;
                     const isCurrentlySelected = targetOption.classList.contains('selected');
                     const allOption = contentDiv.querySelector(`[data-value="${allValue}"]`);

                     if (clickedValue === allValue) {
                         // If 'All' is clicked, clear others and select 'All'
                         selectedSet.clear();
                         selectedSet.add(allValue);
                         contentDiv.querySelectorAll('[data-value]').forEach(opt => opt.classList.remove('selected'));
                         if (allOption) allOption.classList.add('selected');
                     } else {
                         // If a specific item is clicked
                         selectedSet.delete(allValue); // Remove 'All' if it exists
                         if (allOption) allOption.classList.remove('selected');

                         targetOption.classList.toggle('selected'); // Toggle the clicked item
                         if (targetOption.classList.contains('selected')) {
                             selectedSet.add(clickedValue); // Add if selected
                         } else {
                             selectedSet.delete(clickedValue); // Remove if deselected
                         }

                         // If nothing is selected, select 'All'
                         if (selectedSet.size === 0) {
                             selectedSet.add(allValue);
                             if (allOption) allOption.classList.add('selected');
                         }
                     }

                     // Update button text using the helper function
                     this.updateButtonText(selectedSet, buttonElement, allValue);

                     // Trigger subcategory update ONLY if this is the category selector
                     if (triggerSubcategoryUpdate) {
                         this.updateSubcategoryOptions(); // Update subcategories
                     }

                     await this.applyFilters(); // Apply all filters including the change (make sure it's awaited)
                 });

                 // Initial setup: Ensure existing options reflect the set state and button text is correct
                 contentDiv.querySelectorAll('[data-value]').forEach(option => {
                     if (selectedSet.has(option.dataset.value)) {
                         option.classList.add('selected');
                     } else {
                         option.classList.remove('selected');
                     }
                 });
                 this.updateButtonText(selectedSet, buttonElement, allValue);
             };


            // Setup each multi-select using query selectors
            this.setupMultiSelect(
                '.category-option', // Use selector string
                window.selectedCategories,
                'All Categories',
                categoryBtn,
                true // YES, trigger subcategory update from here
            );

            this.setupMultiSelect(
                '.country-option', // Use selector string
                window.selectedCountries,
                'All Countries',
                countryBtn
            );

            this.setupMultiSelect(
                '.state-option', // Use selector string
                window.selectedStates,
                'All States',
                stateBtn
            );

            // Setup subcategory initially - listeners attached via delegation above
            // We still need to call updateButtonText initially for subcategories
             this.updateButtonText(window.selectedSubcategories, subcategoryBtn, 'All Subcategories');
             // Attach the listener for subcategories (it needs its own call to setupMultiSelect)
              this.setupMultiSelect(
                  '.subcategory-option', // Use selector string
                  window.selectedSubcategories,
                  'All Subcategories',
                  subcategoryBtn,
                  false // Do NOT trigger update from subcategory selection
              );


            // Setup other filters (date, sort)
            const filterIds = ['dateFilter', 'sortFilter'];
            filterIds.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    // Remove previous listener if any (safer)
                     element.removeEventListener('change', this.boundApplyFilters);
                     // Store the bound function reference to remove it later if needed
                     this.boundApplyFilters = this.applyFilters.bind(this);
                    element.addEventListener('change', this.boundApplyFilters);
                }
            });

            // Add reset button handler
            const resetButton = document.getElementById('resetFilters');
            if (resetButton) {
                 // Ensure only one listener is attached
                 resetButton.removeEventListener('click', this.boundResetFilters);
                 this.boundResetFilters = this.resetFilters.bind(this); // Bind 'this'
                resetButton.addEventListener('click', this.boundResetFilters);
            }
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
                 console.error("One or more range slider elements not found. Skipping setup.");
                 this.rangeSliderElements = null; // Indicate sliders are not set up
                 return;
            }


            let inputTimeout; // Timeout for debouncing input changes

            // Function to visually fill the slider track
            const fillSlider = (from, to, sliderColor, rangeColor, controlSlider) => {
                const rangeDistance = controlSlider.max - controlSlider.min;
                 // Prevent division by zero if min equals max
                 if (rangeDistance === 0) {
                     controlSlider.style.background = rangeColor; // Fill completely if range is zero
                     return;
                 }
                const fromPosition = from.value - controlSlider.min;
                const toPosition = to.value - controlSlider.min;
                // Calculate percentages safely
                 const fromPercent = (fromPosition / rangeDistance) * 100;
                 const toPercent = (toPosition / rangeDistance) * 100;

                controlSlider.style.background = `linear-gradient(
                    to right,
                    ${sliderColor} 0%,
                    ${sliderColor} ${fromPercent}%,
                    ${rangeColor} ${fromPercent}%,
                    ${rangeColor} ${toPercent}%,
                    ${sliderColor} ${toPercent}%,
                    ${sliderColor} 100%)`;
            };

             // Debounced version of applyFilters specifically for sliders
             const debouncedSliderApplyFilters = debounce(async () => await this.applyFilters(), 350); // Slightly longer debounce for sliders


             // Helper to control the 'from' slider handle
            const controlFromSlider = (fromSlider, toSlider, fromInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                 // Ensure 'from' doesn't cross 'to'
                if (from > to) {
                    fromSlider.value = to; // Snap 'from' to 'to's value
                    // Don't update fromInput here, let the input handler do it if needed
                    return; // Exit early as value didn't really change logically
                }
                fromInput.value = from; // Update input to match slider
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider); // Update visual fill
                 debouncedSliderApplyFilters(); // Apply filters after a delay
            };

             // Helper to control the 'to' slider handle
            const controlToSlider = (fromSlider, toSlider, toInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                 // Ensure 'to' doesn't cross 'from'
                if (to < from) {
                    toSlider.value = from; // Snap 'to' to 'from's value
                    // Don't update toInput here
                    return; // Exit early
                }
                toInput.value = to; // Update input to match slider
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider); // Update visual fill
                 debouncedSliderApplyFilters(); // Apply filters after a delay
            };


            // Helper to get parsed integer values from a pair of sliders
            const getParsedValue = (slider1, slider2) => {
                const val1 = parseInt(slider1.value, 10);
                const val2 = parseInt(slider2.value, 10);
                // Return in [min, max] order regardless of which slider was passed first
                 return [Math.min(val1, val2), Math.max(val1, val2)];
            };

            // Generic handler for slider input events
             const handleSliderInput = (e) => {
                 const slider = e.target;
                 const isFromSlider = slider.id.includes('From');
                 // Identify the corresponding elements based on slider group (pledged, goal, raised)
                 const groupPrefix = slider.id.replace('FromSlider', '').replace('ToSlider', '');
                 const fromS = document.getElementById(`${groupPrefix}FromSlider`);
                 const toS = document.getElementById(`${groupPrefix}ToSlider`);
                 const fromI = document.getElementById(`${groupPrefix}FromInput`);
                 const toI = document.getElementById(`${groupPrefix}ToInput`);

                 if (isFromSlider) {
                     controlFromSlider(fromS, toS, fromI);
                 } else {
                     controlToSlider(fromS, toS, toI);
                 }
             };

            // Generic handler for number input changes (debounced)
            const handleNumberInput = (e) => {
                 clearTimeout(inputTimeout);
                 inputTimeout = setTimeout(async () => { // Make async
                    const input = e.target;
                    const isFromInput = input.id.includes('From');
                    const groupPrefix = input.id.replace('FromInput', '').replace('ToInput', '');
                    const fromS = document.getElementById(`${groupPrefix}FromSlider`);
                    const toS = document.getElementById(`${groupPrefix}ToSlider`);
                    const fromI = document.getElementById(`${groupPrefix}FromInput`);
                    const toI = document.getElementById(`${groupPrefix}ToInput`);

                    let value = parseInt(input.value, 10);
                    const minAllowed = parseInt(input.min, 10);
                    const maxAllowed = parseInt(input.max, 10);

                    // Basic validation
                     if (isNaN(value)) {
                         value = isFromInput ? minAllowed : maxAllowed; // Default to min/max if invalid
                     }
                     value = Math.max(minAllowed, Math.min(maxAllowed, value)); // Clamp within min/max attributes

                    // Cross-validation with the other input/slider
                     if (isFromInput) {
                         const maxValue = parseInt(toS.value, 10);
                         if (value > maxValue) value = maxValue; // Don't allow 'from' to exceed 'to'
                         fromS.value = value; // Update slider
                         input.value = value; // Update input (might have been clamped)
                         controlFromSlider(fromS, toS, fromI); // Update fill and trigger filter (debounced)
                     } else { // is To Input
                         const minValue = parseInt(fromS.value, 10);
                         if (value < minValue) value = minValue; // Don't allow 'to' to go below 'from'
                         toS.value = value; // Update slider
                         input.value = value; // Update input
                         controlToSlider(fromS, toS, toI); // Update fill and trigger filter (debounced)
                     }
                     // No need to call applyFilters here, control*Slider handles it (debounced)
                 }, 750); // Debounce input validation slightly longer
            };

            // Generic handler for Enter key press or blur on number inputs
             const handleInputValidationTrigger = (e) => {
                 // Trigger immediate validation only on Enter or blur
                 if (e.type === 'keydown' && e.key !== 'Enter') return;

                 clearTimeout(inputTimeout); // Clear any pending debounced update
                 // Immediately validate and potentially update slider and filters
                 const input = e.target;
                 const isFromInput = input.id.includes('From');
                 const groupPrefix = input.id.replace('FromInput', '').replace('ToInput', '');
                 const fromS = document.getElementById(`${groupPrefix}FromSlider`);
                 const toS = document.getElementById(`${groupPrefix}ToSlider`);
                 const fromI = document.getElementById(`${groupPrefix}FromInput`);
                 const toI = document.getElementById(`${groupPrefix}ToInput`);

                  let value = parseInt(input.value, 10);
                  const minAllowed = parseInt(input.min, 10);
                  const maxAllowed = parseInt(input.max, 10);

                  if (isNaN(value)) {
                       value = isFromInput ? minAllowed : maxAllowed;
                  }
                  value = Math.max(minAllowed, Math.min(maxAllowed, value));

                  if (isFromInput) {
                       const maxValue = parseInt(toS.value, 10);
                       if (value > maxValue) value = maxValue;
                       fromS.value = value;
                       input.value = value; // Ensure input reflects validated value
                       controlFromSlider(fromS, toS, fromI); // Update visuals
                  } else {
                       const minValue = parseInt(fromS.value, 10);
                       if (value < minValue) value = minValue;
                       toS.value = value;
                       input.value = value; // Ensure input reflects validated value
                       controlToSlider(fromS, toS, toI); // Update visuals
                  }
                  // Explicitly apply filters immediately on Enter/Blur after validation
                   this.applyFilters(); // Await not strictly needed here
             };


            // Attach listeners using the generic handlers
            const sliders = [fromSlider, toSlider, goalFromSlider, goalToSlider, raisedFromSlider, raisedToSlider];
            const inputs = [fromInput, toInput, goalFromInput, goalToInput, raisedFromInput, raisedToInput];

            sliders.forEach(slider => {
                slider.addEventListener('input', handleSliderInput);
            });

            inputs.forEach(input => {
                input.addEventListener('input', handleNumberInput); // Debounced update on typing
                 input.addEventListener('keydown', handleInputValidationTrigger); // Immediate update on Enter
                 input.addEventListener('blur', handleInputValidationTrigger); // Immediate update on blur
            });


            // Store references for reset function IF setup was successful
             this.rangeSliderElements = {
                 fromSlider, toSlider, fromInput, toInput,
                 goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                 raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                 fillSlider // Keep reference to the helper
             };

            // Initial visual setup for all sliders
            if (this.rangeSliderElements) {
                 fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                 fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);
                 fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
            }
        }


        // NEW METHOD to update subcategory options
        updateSubcategoryOptions() {
            const selectedCategories = window.selectedCategories || new Set(['All Categories']);
            const subcategoryMap = window.categorySubcategoryMap || {};
            const subcategoryOptionsContainer = document.getElementById('subcategoryOptionsContainer');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

             if (!subcategoryOptionsContainer || !subcategoryBtn) {
                  console.error("Subcategory container or button not found.");
                  return; // Cannot update if elements are missing
             }

            let availableSubcategories = new Set();
            let isAllCategoriesSelected = selectedCategories.has('All Categories');

            // Determine which subcategories to show
            if (isAllCategoriesSelected || selectedCategories.size === 0) {
                 // If 'All Categories' is selected or no category selected, use the special 'All Categories' list from the map
                 (subcategoryMap['All Categories'] || ['All Subcategories']).forEach(subcat => availableSubcategories.add(subcat));
            } else {
                // Otherwise, collect subcategories from the specifically selected categories
                 availableSubcategories.add('All Subcategories'); // Always include 'All Subcategories'
                 selectedCategories.forEach(cat => {
                     // Ensure we access the map correctly and handle potential undefined categories
                     (subcategoryMap[cat] || []).forEach(subcat => {
                         if (subcat && subcat !== 'All Subcategories') { // Check subcat is valid and avoid adding 'All' twice
                            availableSubcategories.add(subcat);
                         }
                     });
                 });
            }


            // Sort the subcategories (keeping 'All Subcategories' first)
            const sortedSubcategories = Array.from(availableSubcategories);
            sortedSubcategories.sort((a, b) => {
                if (a === 'All Subcategories') return -1;
                if (b === 'All Subcategories') return 1;
                // LocaleCompare is good for string sorting
                 if (a && b) return a.localeCompare(b);
                 return 0; // Handle potential null/undefined if data is messy
            });

            // --- Efficiently update options ---
            const currentOptions = new Map();
             subcategoryOptionsContainer.querySelectorAll('.subcategory-option').forEach(opt => {
                 currentOptions.set(opt.dataset.value, opt);
             });

             let lastAppendedNode = null; // Keep track for inserting new nodes

             // Iterate through sorted list, update existing or add new
             sortedSubcategories.forEach(subcatValue => {
                 if (currentOptions.has(subcatValue)) {
                     // Option exists, ensure it's in the right place relative to the last node
                     const existingNode = currentOptions.get(subcatValue);
                     if (lastAppendedNode && lastAppendedNode.nextSibling !== existingNode) {
                         subcategoryOptionsContainer.insertBefore(existingNode, lastAppendedNode.nextSibling);
                     }
                     lastAppendedNode = existingNode;
                     currentOptions.delete(subcatValue); // Mark as processed
                 } else {
                     // Option is new, create and insert it
                     const newOption = document.createElement('div');
                     newOption.className = 'subcategory-option';
                     newOption.dataset.value = subcatValue;
                     newOption.textContent = subcatValue;
                      // Add special class for styling 'All' option
                      if (subcatValue === 'All Subcategories') {
                           newOption.classList.add('all-subcategories-option'); // Add a class if needed for styling break etc.
                      }
                     subcategoryOptionsContainer.insertBefore(newOption, lastAppendedNode ? lastAppendedNode.nextSibling : subcategoryOptionsContainer.firstChild);
                     lastAppendedNode = newOption;
                 }
             });

             // Remove any old options that are no longer needed
             currentOptions.forEach(oldNode => {
                 subcategoryOptionsContainer.removeChild(oldNode);
             });
            // --- End efficient update ---


            // Reset current subcategory selection to 'All Subcategories' in the JS Set
            window.selectedSubcategories = new Set(['All Subcategories']);

             // Update the 'selected' class in the DOM
             subcategoryOptionsContainer.querySelectorAll('.subcategory-option').forEach(opt => {
                 opt.classList.toggle('selected', opt.dataset.value === 'All Subcategories');
             });

            // Update the subcategory button text
            this.updateButtonText(window.selectedSubcategories, subcategoryBtn, 'All Subcategories');

             // Re-attach/confirm event listeners are handled by delegation set up in setupFilters
             // No need to call setupMultiSelect here again if delegation is used.
        }

    } // End of TableManager class

    function onRender(event) {
        if (!window.rendered) {
            window.tableManager = new TableManager(); // Creates and initializes the table
            window.rendered = true;

            // Add resize observer for dynamic height adjustment
            const resizeObserver = new ResizeObserver(() => {
                if (window.tableManager) {
                    // Debounce adjustHeight slightly for resize events
                    clearTimeout(window.resizeTimeout);
                    window.resizeTimeout = setTimeout(() => {
                         window.tableManager.adjustHeight();
                    }, 100); // Increased debounce for resize
                }
            });
            // Observe the main component wrapper or a suitable parent element
            const componentRoot = document.body; // Observe body or a specific container if available
            if (componentRoot) {
                 resizeObserver.observe(componentRoot);
            } else {
                 console.error("Component root not found for ResizeObserver.");
            }
        } else {
             // Handle potential re-renders if necessary (e.g., Streamlit updates component args)
             // For now, just ensure height is adjusted on re-render
             if (window.tableManager) {
                  window.tableManager.adjustHeight();
             }
        }
    }

    // Initial setup listeners
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
"""

# Create and use the component
table_component = generate_component('searchable_table', template=css + template, script=script)
# Add a key for stability if needed (especially if component args change)
table_component(key="kickstarter_table_v2")