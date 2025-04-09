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
import reverse_geocode # Import reverse_geocode

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
             # Ensure 'All Subcategories' is first
             all_sorted_subcats = sorted(list(all_subcategories_set))
             category_subcategory_map['All Categories'] = ['All Subcategories'] + all_sorted_subcats


             # Sort subcategories within each category
             for cat in category_subcategory_map:
                 # Keep 'All Subcategories' first if present, sort the rest
                 subcats = category_subcategory_map[cat]
                 prefix = []
                 rest = []
                 if 'All Subcategories' in subcats:
                     prefix = ['All Subcategories']
                     rest = sorted([s for s in subcats if s != 'All Subcategories'])
                 else:
                      rest = sorted(subcats) # Sort all if 'All Subcategories' isn't there

                 # Rebuild the list for the category
                 category_subcategory_map[cat] = prefix + rest

        # Handle Subcategories if Category doesn't exist (fallback)
        elif 'Subcategory' in _lf.schema and 'Category' not in _lf.schema:
             subcategories_unique = _lf.select(pl.col('Subcategory')).unique().collect()['Subcategory']
             all_subcats = sorted(subcategories_unique.filter(subcategories_unique.is_not_null() & (subcategories_unique != "N/A")).to_list())
             category_subcategory_map['All Categories'] = ['All Subcategories'] + all_subcats # Add to default

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
                  extracted_states = states_collected.str.extract(r'>(\w+)<', 1).unique().drop_nulls().to_list() # Extract content
                  options['states'] += sorted([state.capitalize() for state in extracted_states if state.lower() != 'unknown']) # Capitalize and filter unknown
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
        max_raised = int(max_raised_calc_val) if max_raised_calc_val is not None else 500 # Use default if max is None
        print("Min/max ranges calculated.")
    except Exception as e:
        st.error(f"Error calculating min/max filter ranges: {e}. Using defaults.")
else:
    st.warning("Missing columns required for min/max filter ranges in schema. Using defaults.")


# --- Load Precomputed Country Distances ---
@st.cache_data
def load_country_distances() -> pl.DataFrame | None:
    path = 'country_distances.parquet'
    if not os.path.exists(path):
        st.error(f"{path} not found. Please run compute_country_distances.py first.")
        return None
    try:
        print("Loading precomputed country distances...")
        df_dist = pl.read_parquet(path)
        print("Country distances loaded.")
        return df_dist
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

df_country_distances = load_country_distances()

# --- Geolocation Fetching & Country Code Determination ---
loc = get_geolocation()
user_location = None
user_country_code = None

if loc and 'coords' in loc:
    try:
        coords = (loc['coords']['latitude'], loc['coords']['longitude'])
        # Add a spinner while reverse geocoding
        with st.spinner('Determining your country...'):
            geo_info_list = reverse_geocode.get(coords) # Returns a list
            if geo_info_list: # Check if the list is not empty
                 geo_info = geo_info_list # Take the first result
                 user_country_code = geo_info.get('country_code')
                 print(f"User country code determined: {user_country_code}")
                 # Still store precise location for potential future use / JS check
                 user_location = {
                     'latitude': loc['coords']['latitude'],
                     'longitude': loc['coords']['longitude'],
                     'country_code': user_country_code # Add country code here
                 }
                 loading_success = st.success(f"Location ({geo_info.get('city', 'N/A')}, {user_country_code}) received!")
                 time.sleep(1.5)
                 loading_success.empty()
            else:
                 st.warning("Could not determine country from coordinates.")
                 # Keep user_location with just lat/lon if reverse geocoding fails
                 user_location = {
                     'latitude': loc['coords']['latitude'],
                     'longitude': loc['coords']['longitude']
                 }

    except Exception as e:
        st.error(f"Error during reverse geocoding: {e}")
         # Keep user_location with just lat/lon if reverse geocoding fails
        user_location = {
            'latitude': loc['coords']['latitude'],
            'longitude': loc['coords']['longitude']
        }

# --- Assign Distance based on Country Code (Lazy) ---
if user_country_code and df_country_distances is not None and 'Country Code' in lf.schema:
    print(f"Assigning distances based on user country: {user_country_code}")
    # Filter precomputed distances for the user's country
    user_distances = df_country_distances.filter(pl.col('code_from') == user_country_code) \
                                         .select(['code_to', 'distance']) \
                                         .rename({'code_to': 'Country Code', 'distance': 'Distance'}) # Rename for join

    # Join the main LazyFrame with the filtered distances
    lf = lf.join(
        user_distances.lazy(),
        on='Country Code',
        how='left' # Keep all projects, assign null distance if country code doesn't match
    )

    # Fill missing distances with infinity and ensure correct type
    lf = lf.with_columns(
        pl.col('Distance').fill_null(float('inf')).cast(pl.Float64)
    )
    print("Country-based distances assigned to LazyFrame plan.")

else:
    if not user_country_code:
         print("User country code not available.")
    if df_country_distances is None:
         print("Precomputed country distances not loaded.")
    if 'Country Code' not in lf.schema:
         print("'Country Code' column missing in main data schema.")

    print("Setting Distance to infinity (Lazy).")
    # Add Distance column lazily if it wasn't added/filled above
    if 'Distance' not in lf.schema:
        lf = lf.with_columns(pl.lit(float('inf')).cast(pl.Float64).alias('Distance'))
    else:
         # Ensure it's float and fill nulls if join failed partially
         lf = lf.with_columns(pl.col('Distance').fill_null(float('inf')).cast(pl.Float64))


# --- Main Collect and Cleanup ---
df_collected = None
try:
    print("Collecting final DataFrame for display...")
    start_collect_time = time.time()
    # Check if Distance column exists before collecting
    if 'Distance' not in lf.columns:
         st.error("Critical Error: Distance column was not added to the LazyFrame before collection.")
         # Attempt to add it now as a fallback, though ideally it should exist
         lf = lf.with_columns(pl.lit(float('inf')).cast(pl.Float64).alias('Distance'))

    df_collected = lf.collect(streaming=True)
    collect_duration = time.time() - start_collect_time
    # Check if Distance column actually made it into the collected frame
    if 'Distance' not in df_collected.columns:
        st.error("Critical Error: Distance column is missing in the collected DataFrame.")
        # Add it here with default values if absolutely necessary, but indicates a prior logic error
        df_collected = df_collected.with_columns(pl.lit(float('inf')).cast(pl.Float64).alias('Distance'))

    print(f"Collect duration: {collect_duration:.2f} seconds")
    loaded = st.success(f"Loaded {len(df_collected)} projects successfully!")
    time.sleep(1.5)
    loaded.empty()
except Exception as e:
    st.error(f"Error collecting final DataFrame: {e}")
    df_collected = pl.DataFrame() # Ensure it's an empty Polars DF
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
     # Check if the original lf was likely empty vs collection failure
     if lf.head(1).collect().is_empty():
         st.warning("Original data source seems empty or filtered to empty.")
     else:
         st.error("Data collection resulted in an empty DataFrame or failed. Cannot display table.")
     st.stop()


# --- Generate Table HTML (using collected DataFrame) ---
def generate_table_html(df_display: pl.DataFrame): # Expect Eager DataFrame
    # Define visible columns
    visible_columns = ['Project Name', 'Creator', 'Pledged Amount', 'Link', 'Country', 'State']

    # Ensure all required columns for data attributes exist (Distance should be present now)
    required_data_cols = [
        'Category', 'Subcategory', 'Raw Pledged', 'Raw Goal', 'Raw Raised',
        'Raw Date', 'Raw Deadline', 'Backer Count', 'latitude', 'longitude',
        'Country Code', 'Distance', 'Popularity Score' # Distance is crucial
    ]
    missing_data_cols = [col for col in required_data_cols if col not in df_display.columns]
    if missing_data_cols:
        # Make this a more severe error as Distance is fundamental now
        st.error(f"FATAL: Missing required columns for table generation: {missing_data_cols}. Check data processing steps, especially Distance assignment.")
        # Add a dummy Distance column to prevent crashing html generation, but data will be wrong
        if 'Distance' in missing_data_cols and 'Distance' not in df_display.columns:
             df_display = df_display.with_columns(pl.lit(float('inf')).alias('Distance'))
        # return "", "" # Stop generation if critical columns missing

    # Ensure visible columns exist
    missing_visible_cols = [col for col in visible_columns if col not in df_display.columns]
    if missing_visible_cols:
         st.warning(f"Missing visible columns for table: {missing_visible_cols}. Check database_download.py or initial data processing.")
         # Attempt to continue with available columns
         visible_columns = [col for col in visible_columns if col in df_display.columns]
         if not visible_columns:
              return "", "" # Cannot proceed if no visible columns


    # Generate header for visible columns only
    header_html = ''.join(f'<th scope="col">{column}</th>' for column in visible_columns)

    # Generate table rows with raw values in data attributes
    rows_html = ''
    # Convert collected DataFrame to dicts efficiently
    # Add error handling for to_dicts()
    try:
        data_dicts = df_display.to_dicts()
    except Exception as e:
        st.error(f"Error converting DataFrame to dictionaries: {e}")
        return header_html, "" # Return header but empty rows


    for row in data_dicts:
         # Safely format data attributes (Check Distance exists in the row dict)
        distance_val = row.get('Distance', float('inf'))
        # Ensure distance is finite for formatting, default to large number if inf
        display_distance = distance_val if np.isfinite(distance_val) else 9999999.0

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
            data-distance="{display_distance:.2f}"
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
                 # Ensure value is a string before adding
                 state_html = str(value) if value is not None else 'N/A'
                 visible_cells += f'<td>{state_html}</td>'
            else:
                # Escape potential HTML in other cells
                import html
                visible_cells += f'<td>{html.escape(str(value))}</td>'

        rows_html += f'<tr class="table-row" {data_attrs}>{visible_cells}</tr>'

    return header_html, rows_html

# Generate HTML using the collected DataFrame
header_html, rows_html = generate_table_html(df_collected)

# --- HTML Template ---
# RE-ADD user location (including country code if available)
# RE-ADD category-subcategory mapping
# REMOVE conditional for 'Near Me' sort option
template = f"""
<script>
    window.userLocation = {json.dumps(user_location) if user_location else 'null'};
    window.hasLocation = {json.dumps(bool(user_location))};
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
# No changes needed here, as the 'nearme' sort uses data-distance,
# which is now populated with country-level distance by Python.
# The check `if (!this.userLocation)` still correctly handles the
# case where location permission wasn't granted.
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

            await this.applyAllFilters();
        }

        initialize() {
            this.setupSearchAndPagination();
            this.setupFilters(); // This will now handle the hierarchical setup
            this.setupRangeSlider();
            this.currentSort = 'popularity';  // Set default sort to popularity
            this.applyAllFilters();
            this.updateTable();
            // Initial population of subcategories based on default 'All Categories'
            this.updateSubcategoryOptions(); 
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
            let state = '';
            if (stateCell) {
                for (const cls of stateCell.classList) {
                    if (cls.startsWith('state-')) {
                        state = cls.substring(6);
                        break;
                    }
                }
            }
            if (!filters.states.includes('All States')) {
                const stateLower = state.toLowerCase();
                const matchingState = filters.states.find(s => s.toLowerCase() === stateLower);
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

            // Reset subcategory selections (will be repopulated by updateSubcategoryOptions)
            const subcategoryOptionsContainer = document.getElementById('subcategoryOptionsContainer');
            subcategoryOptionsContainer.innerHTML = ''; // Clear existing options
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');
            if (window.selectedSubcategories) window.selectedSubcategories.clear();
            if (window.selectedSubcategories) window.selectedSubcategories.add('All Subcategories');
            // Update subcategory options based on the reset category ('All Categories')
            this.updateSubcategoryOptions(); 

            // Reset the stored selections in the Sets
            if (window.selectedCategories) window.selectedCategories.clear();
            if (window.selectedCountries) window.selectedCountries.clear();
            if (window.selectedStates) window.selectedStates.clear();
            // Subcategories reset done above

            // Re-add "All" options to the Sets
            if (window.selectedCategories) window.selectedCategories.add('All Categories');
            if (window.selectedCountries) window.selectedCountries.add('All Countries');
            if (window.selectedStates) window.selectedStates.add('All States');
            // Subcategories reset done above

            // Reset ranges, search, sort etc. as before ...
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

            // Reset Date Filter
            const dateFilterSelect = document.getElementById('dateFilter');
            if (dateFilterSelect) {
                dateFilterSelect.value = 'All Time'; // Set to the default value
            }
            this.searchInput.value = '';
            this.currentSearchTerm = '';
            this.currentFilters = null; // Filters will be reapplied by applyAllFilters
            this.currentSort = 'popularity';
            document.getElementById('sortFilter').value = 'popularity';
            this.visibleRows = this.allRows;
            this.applyAllFilters(); // Apply filters which includes the reset subcategory
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
                 // Check if the only selected item is the 'All' option
                 if (selectedArray.length === 1 && selectedArray[0] === allValueLabel) {
                     buttonElement.textContent = allValueLabel;
                 } else if (selectedArray.length === 0 ) { // Handle case where set might be empty temporarily
                     buttonElement.textContent = allValueLabel; // Default to 'All'
                 }
                 else {
                     // Filter out the 'All' option if other items are selected
                     const displayItems = selectedArray.filter(item => item !== allValueLabel);
                     const sortedArray = displayItems.sort((a, b) => a.localeCompare(b));

                     if (sortedArray.length === 0) { // Should not happen if logic is correct, but safe fallback
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
            this.setupMultiSelect = (options, selectedSet, allValue, buttonElement, triggerSubcategoryUpdate = false) => {
                const allOption = Array.from(options).find(opt => opt.dataset.value === allValue); // Safer find

                options.forEach(option => {
                    // Remove existing listener before adding new one (prevents duplicates)
                    option.replaceWith(option.cloneNode(true)); // Simple way to remove listeners
                });

                // Get fresh references after cloning
                 const freshOptions = buttonElement.nextElementSibling.querySelectorAll('[data-value]');
                 const freshAllOption = Array.from(freshOptions).find(opt => opt.dataset.value === allValue);

                freshOptions.forEach(option => {
                     // Add selected class if it's in the current set
                     if (selectedSet.has(option.dataset.value)) {
                         option.classList.add('selected');
                     }

                     option.addEventListener('click', (e) => {
                        const clickedValue = e.target.dataset.value;
                        const isCurrentlySelected = e.target.classList.contains('selected');

                        if (clickedValue === allValue) {
                            // If 'All' is clicked, clear others and select 'All'
                            selectedSet.clear();
                            selectedSet.add(allValue);
                            freshOptions.forEach(opt => opt.classList.remove('selected'));
                            if(freshAllOption) freshAllOption.classList.add('selected');
                        } else {
                            // If a specific item is clicked
                            selectedSet.delete(allValue); // Remove 'All' if it exists
                            if(freshAllOption) freshAllOption.classList.remove('selected');

                            e.target.classList.toggle('selected'); // Toggle the clicked item
                            if (e.target.classList.contains('selected')) {
                                selectedSet.add(clickedValue); // Add if selected
                            } else {
                                selectedSet.delete(clickedValue); // Remove if deselected
                            }

                            // If nothing is selected, select 'All'
                            if (selectedSet.size === 0) {
                                selectedSet.add(allValue);
                                if(freshAllOption) freshAllOption.classList.add('selected');
                            }
                        }

                        // Update button text using the helper function
                        this.updateButtonText(selectedSet, buttonElement, allValue);

                        // Trigger subcategory update ONLY if this is the category selector
                        if (triggerSubcategoryUpdate) {
                            this.updateSubcategoryOptions(); // Update subcategories
                        }

                        this.applyFilters(); // Apply all filters including the change
                    });
                });

                // Initialize button text correctly
                this.updateButtonText(selectedSet, buttonElement, allValue);
            };

            // Setup each multi-select
            this.setupMultiSelect(
                document.querySelectorAll('.category-option'),
                window.selectedCategories,
                'All Categories',
                categoryBtn,
                true // YES, trigger subcategory update from here
            );

            this.setupMultiSelect(
                document.querySelectorAll('.country-option'),
                window.selectedCountries,
                'All Countries',
                countryBtn
            );

            this.setupMultiSelect(
                document.querySelectorAll('.state-option'),
                window.selectedStates,
                'All States',
                stateBtn
            );

            // Setup subcategory initially (listeners will be re-attached by updateSubcategoryOptions)
             this.setupMultiSelect(
                 document.querySelectorAll('.subcategory-option'),
                 window.selectedSubcategories,
                 'All Subcategories',
                 subcategoryBtn
             );


            // Setup other filters (date, sort)
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

            // Range slider initialization remains the same
            // this.setupRangeSlider(); // Called in initialize()
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
                // Check if fromInput exists before trying to access its value
                const currentFromInputValue = fromInput ? parseInt(fromInput.value) : from; 
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                // Ensure the slider doesn't go past the 'to' slider
                if (from > to) {
                    fromSlider.value = to;
                    if (fromInput) fromInput.value = to; // Update input if it exists
                } else {
                     if (fromInput) fromInput.value = from; // Update input if it exists
                }
            };
            
            const controlToSlider = (fromSlider, toSlider, toInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                 // Check if toInput exists before trying to access its value
                const currentToInputValue = toInput ? parseInt(toInput.value) : to; 
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                // Ensure the slider doesn't go below the 'from' slider
                if (from <= to) {
                    // toSlider.value = to; // Slider value already set by input event or validateAndUpdateRange
                    if (toInput) toInput.value = to; // Update input if it exists
                } else {
                    if (toInput) toInput.value = from; // Update input if it exists
                    toSlider.value = from; // Adjust slider position
                }
            };

            const getParsedValue = (fromSlider, toSlider) => {
                // Ensure sliders exist before getting value
                const from = fromSlider ? parseInt(fromSlider.value) : 0;
                const to = toSlider ? parseInt(toSlider.value) : 0;
                return [from, to];
            };

            const validateAndUpdateRange = (input, isMin = true, immediate = false) => {
                 const updateValues = () => {
                     // Ensure the input element exists before proceeding
                     if (!input) {
                         console.error("validateAndUpdateRange called with null input");
                         return;
                     }
            
                     let value = parseInt(input.value);
                     const minAllowed = parseInt(input.min);
                     const maxAllowed = parseInt(input.max);
            
                     // Determine which set of sliders/inputs we are working with
                     let relevantFromSlider, relevantToSlider, relevantFromInput, relevantToInput;
                     if (input.id.startsWith('goal')) {
                         relevantFromSlider = goalFromSlider;
                         relevantToSlider = goalToSlider;
                         relevantFromInput = goalFromInput;
                         relevantToInput = goalToInput;
                     } else if (input.id.startsWith('raised')) {
                         relevantFromSlider = raisedFromSlider;
                         relevantToSlider = raisedToSlider;
                         relevantFromInput = raisedFromInput;
                         relevantToInput = raisedToInput;
                     } else { // Pledged
                         relevantFromSlider = fromSlider;
                         relevantToSlider = toSlider;
                         relevantFromInput = fromInput;
                         relevantToInput = toInput;
                     }
            
                     // Ensure relevant sliders exist before proceeding
                     if (!relevantFromSlider || !relevantToSlider) {
                         console.error("validateAndUpdateRange could not find relevant sliders for input:", input.id);
                         return;
                     }

                     if (isNaN(value)) {
                         value = isMin ? minAllowed : maxAllowed;
                     }
            
                     if (isMin) {
                         const maxValue = parseInt(relevantToSlider.value);
                         value = Math.max(minAllowed, Math.min(maxValue, value));
                         relevantFromSlider.value = value;
                         input.value = value; // Update the input itself in case of clamping
                         // NOW, also update the slider visuals by calling the control function
                         controlFromSlider(relevantFromSlider, relevantToSlider, relevantFromInput); 
                     } else {
                         const minValue = parseInt(relevantFromSlider.value);
                         value = Math.max(minValue, Math.min(maxAllowed, value));
                         relevantToSlider.value = value;
                         input.value = value; // Update the input itself in case of clamping
                         // NOW, also update the slider visuals by calling the control function
                         controlToSlider(relevantFromSlider, relevantToSlider, relevantToInput); 
                     }
            
                     // Note: fillSlider is called within controlFromSlider/controlToSlider, 
                     // so no separate call is needed here.
                     debouncedApplyFilters();
                 };
            
                 if (immediate) {
                     clearTimeout(inputTimeout);
                     updateValues();
                 } else {
                     clearTimeout(inputTimeout);
                     // Reduced timeout for quicker visual feedback from typing
                     inputTimeout = setTimeout(updateValues, 500); 
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

        // NEW METHOD to update subcategory options
        updateSubcategoryOptions() {
            const selectedCategories = window.selectedCategories || new Set(['All Categories']);
            const subcategoryMap = window.categorySubcategoryMap || {};
            const subcategoryOptionsContainer = document.getElementById('subcategoryOptionsContainer');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            let availableSubcategories = new Set();
            let isAllCategoriesSelected = selectedCategories.has('All Categories');

            // Determine which subcategories to show
            if (isAllCategoriesSelected || selectedCategories.size === 0) {
                 // If 'All Categories' is selected or no category is selected, show all subcategories
                 (subcategoryMap['All Categories'] || []).forEach(subcat => availableSubcategories.add(subcat));
            } else {
                // Otherwise, collect subcategories from the specifically selected categories
                 availableSubcategories.add('All Subcategories'); // Always include 'All Subcategories'
                 selectedCategories.forEach(cat => {
                     (subcategoryMap[cat] || []).forEach(subcat => {
                         if (subcat !== 'All Subcategories') { // Avoid adding it twice
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
                return a.localeCompare(b);
            });

            // Generate HTML for new options
            subcategoryOptionsContainer.innerHTML = sortedSubcategories.map(opt =>
                `<div class="subcategory-option" data-value="${opt}">${opt}</div>`
            ).join('');

            // Reset current subcategory selection to 'All Subcategories'
            window.selectedSubcategories = new Set(['All Subcategories']);
            const allSubcatOption = subcategoryOptionsContainer.querySelector('.subcategory-option[data-value="All Subcategories"]');
            if (allSubcatOption) {
                allSubcatOption.classList.add('selected');
            }
             this.updateButtonText(window.selectedSubcategories, subcategoryBtn, 'All Subcategories'); // Use helper


            // Re-attach event listeners to the *new* subcategory options
            this.setupMultiSelect(
                subcategoryOptionsContainer.querySelectorAll('.subcategory-option'),
                window.selectedSubcategories,
                'All Subcategories',
                subcategoryBtn,
                false // Do not trigger updateSubcategoryOptions from subcategory selection
            );
        }
    }

    function onRender(event) {
        if (!window.rendered) {
            window.tableManager = new TableManager();
            window.rendered = true;

            // Add resize observer
            const resizeObserver = new ResizeObserver(() => {
                if (window.tableManager) {
                    // Debounce adjustHeight slightly for resize events
                    clearTimeout(window.resizeTimeout);
                    window.resizeTimeout = setTimeout(() => {
                         window.tableManager.adjustHeight();
                    }, 50);
                }
            });
            const tableWrapper = document.querySelector('.table-wrapper');
            if (tableWrapper) {
                 resizeObserver.observe(tableWrapper);
            } else {
                 console.error("Table wrapper not found for ResizeObserver.");
            }
        } else {
             // Handle potential re-renders if necessary, maybe re-adjust height
             if (window.tableManager) {
                  window.tableManager.adjustHeight();
             }
        }
    }
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
"""

# Create and use the component
table_component = generate_component('searchable_table', template=css + template, script=script)
table_component(key="kickstarter_table")