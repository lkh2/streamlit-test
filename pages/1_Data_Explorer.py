import time
import streamlit as st
import streamlit.components.v1 as components
import tempfile, os
import json
import polars as pl

st.set_page_config(
    layout="wide", 
    page_icon="📊", 
    page_title="Data Explorer",
    initial_sidebar_state="collapsed"
)

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

# *** IMPORTANT: Process the Parquet file before running this script (See README.md) ***
parquet_source_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.parquet")

if not os.path.exists(parquet_source_path):
    st.error(f"Parquet data source not found at '{parquet_source_path}'. Please ensure the file/directory exists in the project root.")
    st.stop()

lf: pl.LazyFrame | None = None 
try:
    print(f"Scanning Parquet source: {parquet_source_path}")
    lf = pl.scan_parquet(parquet_source_path) 
    print("LazyFrame created successfully from source.")
except Exception as e:
     st.error(f"Error scanning the Parquet source '{parquet_source_path}': {e}")
     if hasattr(e, 'context_stack'):
         context_info = getattr(e, 'context_stack', 'Not available') 
         st.error(f"Context stack: {context_info}")
     else:
          st.error("Context stack information not available.") 
     st.stop()

try:
    collected_schema = lf.collect_schema() 
    schema_check = lf.head(0).collect()
    if schema_check.width == 0 :
         st.error(f"Loaded data from '{parquet_source_path}' appears to have no columns or is invalid. Please check the source file/directory.")
         st.stop()
    print("LazyFrame Schema:", collected_schema)
except Exception as e:
     st.error(f"Error during initial data check on '{parquet_source_path}': {e}. Cannot proceed.")
     st.stop()

if 'State' in collected_schema:
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

def get_filter_options(schema_dict: dict, data_lf: pl.LazyFrame):
    print("Calculating filter options...")
    options = {
        'categories': ['All Categories'],
        'countries': ['All Countries'],
        'states': ['All States'],
        'date_ranges': [
            'All Time', 'Last Month', 'Last 6 Months', 'Last Year',
            'Last 5 Years', 'Last 10 Years'
        ]
    }
    category_subcategory_map = {'All Categories': ['All Subcategories']}

    try:
        if 'Category' in schema_dict:
            categories_unique = data_lf.select(pl.col('Category')).unique().collect()['Category']
            valid_categories = sorted(categories_unique.filter(categories_unique.is_not_null() & (categories_unique != "N/A")).to_list())
            options['categories'] += valid_categories
            for cat in valid_categories:
                category_subcategory_map[cat] = []

        if 'Category' in schema_dict and 'Subcategory' in schema_dict:
             cat_subcat_pairs = data_lf.select(['Category', 'Subcategory']).unique().drop_nulls().collect()
             all_subcategories_set = set()
             for row in cat_subcat_pairs.iter_rows(named=True):
                  category = row['Category']
                  subcategory = row['Subcategory']
                  if category and subcategory and category != "N/A" and subcategory != "N/A":
                       if category in category_subcategory_map:
                           category_subcategory_map[category].append(subcategory)
                       all_subcategories_set.add(subcategory)

             all_sorted_subcats = sorted(list(all_subcategories_set))
             category_subcategory_map['All Categories'] = ['All Subcategories'] + all_sorted_subcats

             for cat in category_subcategory_map:
                 subcats = category_subcategory_map[cat]
                 prefix = []
                 rest = []
                 if 'All Subcategories' in subcats:
                     prefix = ['All Subcategories']
                     rest = sorted([s for s in subcats if s != 'All Subcategories'])
                 else:
                      rest = sorted(subcats)
                 category_subcategory_map[cat] = prefix + rest

        elif 'Subcategory' in schema_dict and 'Category' not in schema_dict:
             subcategories_unique = data_lf.select(pl.col('Subcategory')).unique().collect()['Subcategory']
             all_subcats = sorted(subcategories_unique.filter(subcategories_unique.is_not_null() & (subcategories_unique != "N/A")).to_list())
             category_subcategory_map['All Categories'] = ['All Subcategories'] + all_subcats

        if not category_subcategory_map['All Categories']:
             category_subcategory_map['All Categories'] = ['All Subcategories']

        if 'Country' in schema_dict:
             countries_unique = data_lf.select(pl.col('Country')).unique().collect()['Country']
             options['countries'] += sorted(countries_unique.filter(countries_unique.is_not_null() & (countries_unique != "N/A")).to_list())

        if 'State' in schema_dict and schema_dict['State'] == pl.Utf8:
             states_collected = data_lf.select('State').collect()['State']
             sample_state = states_collected.head(1).to_list()
             if sample_state and sample_state[0] and sample_state[0].startswith('<div class="state_cell state-'):
                  extracted_states = states_collected.str.extract(r'>(\w+)<', 1).unique().drop_nulls().to_list()
                  options['states'] += sorted([state.capitalize() for state in extracted_states if state.lower() != 'unknown'])
             else:
                  plain_states = states_collected.filter(states_collected.is_not_null() & (states_collected != "N/A")).unique().to_list()
                  options['states'] += sorted([s.capitalize() for s in plain_states])
        print("Filter options calculated.")

    except Exception as e:
         st.error(f"Error calculating filter options: {e}")
         options = {
             'categories': ['All Categories'], 'countries': ['All Countries'], 'states': ['All States'],
             'date_ranges': ['All Time', 'Last Month', 'Last 6 Months', 'Last Year', 'Last 5 Years', 'Last 10 Years']
         }
         category_subcategory_map = {'All Categories': ['All Subcategories']}

    return options, category_subcategory_map

filter_options, category_subcategory_map = get_filter_options(lf.collect_schema(), lf) 

min_pledged, max_pledged = 0, 1000
min_goal, max_goal = 0, 10000
min_raised, max_raised = 0, 500 

required_minmax_cols = ['Raw Pledged', 'Raw Goal', 'Raw Raised']
if all(col in lf.collect_schema() for col in required_minmax_cols):
    print("Calculating min/max filter ranges...")
    try:
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
        max_raised = int(max_raised_calc_val) if max_raised_calc_val is not None else 500
        print("Min/max ranges calculated.")
    except Exception as e:
        st.error(f"Error calculating min/max filter ranges: {e}. Using defaults.")
else:
    st.warning("Missing columns required for min/max filter ranges in schema. Using defaults.")

print("Schema before final collect:", lf.collect_schema())

df_collected = None
try:
    print("Collecting final DataFrame for display...")
    start_collect_time = time.time()

    df_collected = lf.collect(engine='streaming')
    collect_duration = time.time() - start_collect_time
    print(f"Data collection took {collect_duration:.2f} seconds.")

    loaded = st.success(f"Loaded {len(df_collected)} projects successfully!")
    time.sleep(1.5)
    loaded.empty()
except Exception as e:
    st.error(f"Error collecting final DataFrame: {e}")
    df_collected = pl.DataFrame()

if df_collected.is_empty():
     is_source_empty = False
     try:
         is_source_empty = lf.head(1).collect().is_empty()
     except Exception as check_err:
         print(f"Could not check if source is empty after failed collect: {check_err}")

     if is_source_empty:
         st.warning("Data source is empty. Displaying empty table.")
         try:
             df_collected = lf.head(0).collect()
         except Exception:
             st.error("Could not retrieve schema from empty source. Stopping.")
             st.stop()
     else:
         st.error("Data collection resulted in an empty DataFrame or failed. Cannot display table.")
         st.stop()

def generate_table_html(df_display: pl.DataFrame): 
    visible_columns = ['Project Name', 'Creator', 'Pledged Amount', 'Link', 'Country', 'State']

    required_data_cols = [
        'Category', 'Subcategory', 'Raw Pledged', 'Raw Goal', 'Raw Raised',
        'Raw Date', 'Raw Deadline', 'Backer Count', 'Popularity Score'
    ]
    missing_data_cols = [col for col in required_data_cols if col not in df_display.columns]
    if missing_data_cols:
        st.error(f"FATAL: Missing required columns for table generation: {missing_data_cols}. Check data processing steps.")

    missing_visible_cols = [col for col in visible_columns if col not in df_display.columns]
    if missing_visible_cols:
         st.warning(f"Missing visible columns for table: {missing_visible_cols}. Check database_download.py or initial data processing.")
         visible_columns = [col for col in visible_columns if col in df_display.columns]
         if not visible_columns:
              return "", ""

    header_html = ''.join(f'<th scope="col">{column}</th>' for column in visible_columns)
    rows_html = ''
    try:
        data_dicts = df_display.to_dicts()
    except Exception as e:
        st.error(f"Error converting DataFrame to dictionaries: {e}")
        return header_html, ""

    for row in data_dicts:
        data_attrs = f'''
            data-category="{row.get('Category', 'N/A')}"
            data-subcategory="{row.get('Subcategory', 'N/A')}"
            data-pledged="{row.get('Raw Pledged', 0.0):.2f}"
            data-goal="{row.get('Raw Goal', 0.0):.2f}"
            data-raised="{row.get('Raw Raised', 0.0):.2f}"
            data-date="{row.get('Raw Date').strftime('%Y-%m-%d') if row.get('Raw Date') else 'N/A'}"
            data-deadline="{row.get('Raw Deadline').strftime('%Y-%m-%d') if row.get('Raw Deadline') else 'N/A'}"
            data-backers="{row.get('Backer Count', 0)}"
            data-popularity="{row.get('Popularity Score', 0.0):.6f}"
        '''
        visible_cells = ''
        for col in visible_columns:
            value = row.get(col, 'N/A') 
            if col == 'Link':
                url = str(value) if value else '#'
                display_url = url if len(url) < 60 else url[:57] + '...'
                visible_cells += f'<td><a href="{url}" target="_blank" title="{url}">{display_url}</a></td>'
            elif col == 'Pledged Amount': 
                import html
                try:
                    amount = int(float(value))
                    formatted_value = f"${amount:,}" 
                except (ValueError, TypeError):
                    formatted_value = 'N/A'
                visible_cells += f'<td>{html.escape(formatted_value)}</td>'
            elif col == 'State': 
                 state_html = str(value) if value is not None else 'N/A'
                 visible_cells += f'<td>{state_html}</td>'
            else:
                import html
                visible_cells += f'<td>{html.escape(str(value))}</td>'

        rows_html += f'<tr class="table-row" {data_attrs}>{visible_cells}</tr>'

    return header_html, rows_html

header_html, rows_html = generate_table_html(df_collected)

template = f"""
<script>
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
    th[scope="col"]:nth-child(1) { width: 25%; } 
    th[scope="col"]:nth-child(2) { width: 12.5%; } 
    th[scope="col"]:nth-child(3) { width: 120px; } 
    th[scope="col"]:nth-child(4) { width: 25%; } 
    th[scope="col"]:nth-child(5) { width: 12.5%; } 
    th[scope="col"]:nth-child(6) { width: 120px; } 

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
        display: none;
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

script = """
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
            this.initialize();
            this.resetFilters();
        }

        async sortRows(sortType) {
            if (sortType === 'popularity') {
                this.visibleRows.sort((a, b) => {
                    const scoreA = parseFloat(a.dataset.popularity);
                    const scoreB = parseFloat(b.dataset.popularity);
                    
                    if (isNaN(scoreA)) return 1;
                    if (isNaN(scoreB)) return -1;
                    
                    return scoreB - scoreA;
                });
            } else if (sortType === 'enddate') {
                this.visibleRows.sort((a, b) => {
                    const deadlineA = new Date(a.dataset.deadline);
                    const deadlineB = new Date(b.dataset.deadline);
                    return deadlineB - deadlineA; 
                });
            } else if (sortType === 'mostfunded') {
                this.visibleRows.sort((a, b) => {
                    const pledgedA = parseFloat(a.dataset.pledged);
                    const pledgedB = parseFloat(b.dataset.pledged);
                    return pledgedB - pledgedA; 
                });
            } else if (sortType === 'mostbacked') {
                this.visibleRows.sort((a, b) => {
                    const backersA = parseInt(a.dataset.backers);
                    const backersB = parseInt(b.dataset.backers);
                    return backersB - backersA; 
                });
            } else {
                this.visibleRows.sort((a, b) => {
                    const dateA = new Date(a.dataset.date);
                    const dateB = new Date(b.dataset.date);
                    return sortType === 'newest' ? dateB - dateA : dateA - dateB;
                });
            }

            const tbody = document.querySelector('#data-table tbody');
            this.visibleRows.forEach(row => row.parentNode && row.parentNode.removeChild(row));
            this.visibleRows.forEach(row => tbody.appendChild(row));
            
            this.currentPage = 1;
            this.updateTable();
        }

        async applyAllFilters() {
            let filteredRows = this.allRows;

            if (this.currentSearchTerm) {
                const pattern = createRegexPattern(this.currentSearchTerm);
                filteredRows = filteredRows.filter(row => {
                    const text = row.textContent || row.innerText;
                    return pattern.test(text);
                });
            }

            if (this.currentFilters) {
                filteredRows = filteredRows.filter(row => {
                    return this.matchesFilters(row, this.currentFilters);
                });
            }

            this.visibleRows = filteredRows;

            await this.sortRows(this.currentSort);

            this.currentPage = 1;
            this.updateTable();
        }

        async applyFilters() {
            const selectedCategories = Array.from(document.querySelectorAll('.category-option.selected'))
                .map(option => option.dataset.value);

            const selectedCountries = Array.from(document.querySelectorAll('.country-option.selected'))
                .map(option => option.dataset.value);

            const selectedStates = Array.from(document.querySelectorAll('.state-option.selected'))
                .map(option => option.dataset.value);

            const selectedSubcategories = Array.from(document.querySelectorAll('.subcategory-option.selected'))
                .map(option => option.dataset.value);

            this.currentFilters = {
                categories: selectedCategories,
                subcategories: selectedSubcategories,
                countries: selectedCountries,
                states: selectedStates,
                date: document.getElementById('dateFilter').value
            };

            const sortSelect = document.getElementById('sortFilter');
            this.currentSort = sortSelect ? sortSelect.value : 'popularity';

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
            this.setupFilters(); 
            this.setupRangeSlider();
            this.currentSort = 'popularity'; 

            this.applyAllFilters(); 
            this.updateTable();
            this.updateSubcategoryOptions(); 
        }

        setupSearchAndPagination() {
            const debouncedSearch = debounce((searchTerm) => {
                this.currentSearchTerm = searchTerm;
                this.applyAllFilters();
            }, 300);

            this.searchInput.addEventListener('input', (e) => {
                debouncedSearch(e.target.value.trim().toLowerCase());
            });

            document.getElementById('prev-page').addEventListener('click', () => this.previousPage());
            document.getElementById('next-page').addEventListener('click', () => this.nextPage());
            window.handlePageClick = (page) => this.goToPage(page);
        }

        matchesFilters(row, filters) {
            const category = row.dataset.category;
            if (!filters.categories.includes('All Categories') && !filters.categories.includes(category)) {
                return false;
            }

            const subcategory = row.dataset.subcategory;
            if (!filters.subcategories.includes('All Subcategories') && !filters.subcategories.includes(subcategory)) {
                return false;
            }

            const country = row.querySelector('td:nth-child(5)').textContent.trim();
            if (!filters.countries.includes('All Countries') && !filters.countries.includes(country)) {
                return false;
            }

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

            const pledged = parseFloat(row.dataset.pledged);
            const goal = parseFloat(row.dataset.goal);
            const raised = parseFloat(row.dataset.raised);
            const date = new Date(row.dataset.date);

            const minPledged = parseFloat(document.getElementById('fromInput').value);
            const maxPledged = parseFloat(document.getElementById('toInput').value);
            if (pledged < minPledged || pledged > maxPledged) return false;

            const minGoal = parseFloat(document.getElementById('goalFromInput').value);
            const maxGoal = parseFloat(document.getElementById('goalToInput').value);
            if (goal < minGoal || goal > maxGoal) return false;

            const minRaised = parseFloat(document.getElementById('raisedFromInput').value);
            const maxRaised = parseFloat(document.getElementById('raisedToInput').value);
            const raisedValue = parseFloat(row.dataset.raised);
            
            if (raisedValue === 0 && minRaised > 0) return false;
            if (raisedValue < minRaised || raisedValue > maxRaised) return false;

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
            const categoryOptions = document.querySelectorAll('.category-option');
            categoryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCategoriesOption = document.querySelector('.category-option[data-value="All Categories"]');
            allCategoriesOption.classList.add('selected');
            const categoryBtn = document.querySelector('.multi-select-btn');
            categoryBtn.textContent = 'All Categories';

            const countryOptions = document.querySelectorAll('.country-option');
            countryOptions.forEach(opt => opt.classList.remove('selected'));
            const allCountriesOption = document.querySelector('.country-option[data-value="All Countries"]');
            allCountriesOption.classList.add('selected');
            const countryBtn = countryOptions[0].closest('.multi-select-dropdown').querySelector('.multi-select-btn');
            countryBtn.textContent = 'All Countries';

            const stateOptions = document.querySelectorAll('.state-option');
            stateOptions.forEach(opt => opt.classList.remove('selected'));
            const allStatesOption = document.querySelector('.state-option[data-value="All States"]');
            allStatesOption.classList.add('selected');
            const stateBtn = stateOptions[0].closest('.multi-select-dropdown').querySelector('.multi-select-btn');
            stateBtn.textContent = 'All States';

            const subcategoryOptionsContainer = document.getElementById('subcategoryOptionsContainer');
            subcategoryOptionsContainer.innerHTML = ''; 
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');
            if (window.selectedSubcategories) window.selectedSubcategories.clear();
            if (window.selectedSubcategories) window.selectedSubcategories.add('All Subcategories');
            this.updateSubcategoryOptions(); 

            if (window.selectedCategories) window.selectedCategories.clear();
            if (window.selectedCountries) window.selectedCountries.clear();
            if (window.selectedStates) window.selectedStates.clear();

            if (window.selectedCategories) window.selectedCategories.add('All Categories');
            if (window.selectedCountries) window.selectedCountries.add('All Countries');
            if (window.selectedStates) window.selectedStates.add('All States');

            if (this.rangeSliderElements) {
                const { 
                    fromSlider, toSlider, fromInput, toInput,
                    goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                    raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                    fillSlider 
                } = this.rangeSliderElements;

                fromSlider.value = fromSlider.min;
                toSlider.value = toSlider.max;
                fromInput.value = fromSlider.min;
                toInput.value = toSlider.max;
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);

                goalFromSlider.value = goalFromSlider.min;
                goalToSlider.value = goalToSlider.max;
                goalFromInput.value = goalFromSlider.min;
                goalToInput.value = goalToSlider.max;
                fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);

                raisedFromSlider.value = raisedFromSlider.min;
                raisedToSlider.value = raisedToSlider.max;
                raisedFromInput.value = raisedFromSlider.min;
                raisedToInput.value = raisedToSlider.max;
                fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
            }

            const dateFilterSelect = document.getElementById('dateFilter');
            if (dateFilterSelect) {
                dateFilterSelect.value = 'All Time'; 
            }
            this.searchInput.value = '';
            this.currentSearchTerm = '';
            this.currentFilters = null; 
            this.currentSort = 'popularity';
            document.getElementById('sortFilter').value = 'popularity';
            this.visibleRows = this.allRows;
            this.applyAllFilters(); 
        }

        updateTable() {
            this.allRows.forEach(row => row.style.display = 'none');
            
            const start = (this.currentPage - 1) * this.pageSize;
            const end = Math.min(start + this.pageSize, this.visibleRows.length);
            
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

                const visibleRowCount = this.visibleRows.slice(
                    (this.currentPage - 1) * this.pageSize,
                    this.currentPage * this.pageSize
                ).length;

                const rowHeight = 52;        
                const headerHeight = 60;     
                const controlsHeight = elements.controls.offsetHeight;
                const paginationHeight = elements.pagination.offsetHeight;
                const padding = 40;
                const minTableHeight = 400;  

                const tableContentHeight = (visibleRowCount * rowHeight) + headerHeight;
                const actualTableHeight = Math.max(tableContentHeight, minTableHeight);

                elements.tableContainer.style.height = `${actualTableHeight}px`;
                elements.tableWrapper.style.height = `${actualTableHeight + controlsHeight + paginationHeight}px`;

                const finalHeight = 
                    elements.titleWrapper.offsetHeight +
                    elements.filterWrapper.offsetHeight +
                    actualTableHeight +
                    controlsHeight +
                    paginationHeight +
                    padding;

                if (!this.lastHeight || Math.abs(this.lastHeight - finalHeight) > 10) {
                    this.lastHeight = finalHeight;
                    Streamlit.setFrameHeight(finalHeight);
                }
            });
        }

        setupFilters() {
            window.selectedCategories = new Set(['All Categories']);
            window.selectedCountries = new Set(['All Countries']);
            window.selectedStates = new Set(['All States']);
            window.selectedSubcategories = new Set(['All Subcategories']); 

            const categoryBtn = document.getElementById('categoryFilterBtn');
            const countryBtn = document.getElementById('countryFilterBtn');
            const stateBtn = document.getElementById('stateFilterBtn');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            this.updateButtonText = (selectedItems, buttonElement, allValueLabel) => {
                 if (!buttonElement) return;

                 const selectedArray = Array.from(selectedItems);
                 if (selectedArray.length === 1 && selectedArray[0] === allValueLabel) {
                     buttonElement.textContent = allValueLabel;
                 } else if (selectedArray.length === 0 ) { 
                     buttonElement.textContent = allValueLabel; 
                 }
                 else {
                     const displayItems = selectedArray.filter(item => item !== allValueLabel);
                     const sortedArray = displayItems.sort((a, b) => a.localeCompare(b));

                     if (sortedArray.length === 0) { 
                          buttonElement.textContent = allValueLabel;
                     } else if (sortedArray.length > 2) {
                          buttonElement.textContent = `${sortedArray[0]}, ${sortedArray[1]} +${sortedArray.length - 2}`;
                     } else {
                          buttonElement.textContent = sortedArray.join(', ');
                     }
                 }
            };

            this.setupMultiSelect = (options, selectedSet, allValue, buttonElement, triggerSubcategoryUpdate = false) => {
                const allOption = Array.from(options).find(opt => opt.dataset.value === allValue); 

                options.forEach(option => {
                    option.replaceWith(option.cloneNode(true)); 
                });

                 const freshOptions = buttonElement.nextElementSibling.querySelectorAll('[data-value]');
                 const freshAllOption = Array.from(freshOptions).find(opt => opt.dataset.value === allValue);

                freshOptions.forEach(option => {
                     if (selectedSet.has(option.dataset.value)) {
                         option.classList.add('selected');
                     }

                     option.addEventListener('click', (e) => {
                        const clickedValue = e.target.dataset.value;
                        const isCurrentlySelected = e.target.classList.contains('selected');

                        if (clickedValue === allValue) {
                            selectedSet.clear();
                            selectedSet.add(allValue);
                            freshOptions.forEach(opt => opt.classList.remove('selected'));
                            if(freshAllOption) freshAllOption.classList.add('selected');
                        } else {
                            selectedSet.delete(allValue); 
                            if(freshAllOption) freshAllOption.classList.remove('selected');

                            e.target.classList.toggle('selected'); 
                            if (e.target.classList.contains('selected')) {
                                selectedSet.add(clickedValue); 
                            } else {
                                selectedSet.delete(clickedValue); 
                            }

                            if (selectedSet.size === 0) {
                                selectedSet.add(allValue);
                                if(freshAllOption) freshAllOption.classList.add('selected');
                            }
                        }

                        this.updateButtonText(selectedSet, buttonElement, allValue);

                        if (triggerSubcategoryUpdate) {
                            this.updateSubcategoryOptions(); 
                        }

                        this.applyFilters(); 
                    });
                });

                this.updateButtonText(selectedSet, buttonElement, allValue);
            };

            this.setupMultiSelect(
                document.querySelectorAll('.category-option'),
                window.selectedCategories,
                'All Categories',
                categoryBtn,
                true 
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

             this.setupMultiSelect(
                 document.querySelectorAll('.subcategory-option'),
                 window.selectedSubcategories,
                 'All Subcategories',
                 subcategoryBtn
             );

            const filterIds = ['dateFilter', 'sortFilter'];
            filterIds.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.addEventListener('change', () => this.applyFilters());
                }
            });

            const resetButton = document.getElementById('resetFilters');
            if (resetButton) {
                resetButton.addEventListener('click', () => this.resetFilters());
            }

        }

        setupRangeSlider() {
            const fromSlider = document.getElementById('fromSlider');
            const toSlider = document.getElementById('toSlider');
            const fromInput = document.getElementById('fromInput');
            const toInput = document.getElementById('toInput');

            const goalFromSlider = document.getElementById('goalFromSlider');
            const goalToSlider = document.getElementById('goalToSlider');
            const goalFromInput = document.getElementById('goalFromInput');
            const goalToInput = document.getElementById('goalToInput');

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
                const currentFromInputValue = fromInput ? parseInt(fromInput.value) : from; 
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                if (from > to) {
                    fromSlider.value = to;
                    if (fromInput) fromInput.value = to; 
                } else {
                     if (fromInput) fromInput.value = from; 
                }
            };
            
            const controlToSlider = (fromSlider, toSlider, toInput) => {
                const [from, to] = getParsedValue(fromSlider, toSlider);
                const currentToInputValue = toInput ? parseInt(toInput.value) : to; 
                fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
                if (from <= to) {
                    if (toInput) toInput.value = to; 
                } else {
                    if (toInput) toInput.value = from; 
                    toSlider.value = from; 
                }
            };

            const getParsedValue = (fromSlider, toSlider) => {
                const from = fromSlider ? parseInt(fromSlider.value) : 0;
                const to = toSlider ? parseInt(toSlider.value) : 0;
                return [from, to];
            };

            const validateAndUpdateRange = (input, isMin = true, immediate = false) => {
                 const updateValues = () => {
                     if (!input) {
                         console.error("validateAndUpdateRange called with null input");
                         return;
                     }
            
                     let value = parseInt(input.value);
                     const minAllowed = parseInt(input.min);
                     const maxAllowed = parseInt(input.max);
            
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
                     } else { 
                         relevantFromSlider = fromSlider;
                         relevantToSlider = toSlider;
                         relevantFromInput = fromInput;
                         relevantToInput = toInput;
                     }
            
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
                         input.value = value; 
                         controlFromSlider(relevantFromSlider, relevantToSlider, relevantFromInput); 
                     } else {
                         const minValue = parseInt(relevantFromSlider.value);
                         value = Math.max(minValue, Math.min(maxAllowed, value));
                         relevantToSlider.value = value;
                         input.value = value; 
                         controlToSlider(relevantFromSlider, relevantToSlider, relevantToInput); 
                     }
            
                     debouncedApplyFilters();
                 };
            
                 if (immediate) {
                     clearTimeout(inputTimeout);
                     updateValues();
                 } else {
                     clearTimeout(inputTimeout);
                     inputTimeout = setTimeout(updateValues, 500); 
                 }
            };

            fromSlider.addEventListener('input', (e) => {
                controlFromSlider(fromSlider, toSlider, fromInput);
                debouncedApplyFilters();
            });

            toSlider.addEventListener('input', (e) => {
                controlToSlider(fromSlider, toSlider, toInput);
                debouncedApplyFilters();
            });

            goalFromSlider.addEventListener('input', (e) => {
                controlFromSlider(goalFromSlider, goalToSlider, goalFromInput);
                debouncedApplyFilters();
            });

            goalToSlider.addEventListener('input', (e) => {
                controlToSlider(goalFromSlider, goalToSlider, goalToInput);
                debouncedApplyFilters();
            });

            raisedFromSlider.addEventListener('input', (e) => {
                controlFromSlider(raisedFromSlider, raisedToSlider, raisedFromInput);
                debouncedApplyFilters();
            });

            raisedToSlider.addEventListener('input', (e) => {
                controlToSlider(raisedFromSlider, raisedToSlider, raisedToInput);
                debouncedApplyFilters();
            });

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

            this.rangeSliderElements = {
                fromSlider, toSlider, fromInput, toInput,
                goalFromSlider, goalToSlider, goalFromInput, goalToInput,
                raisedFromSlider, raisedToSlider, raisedFromInput, raisedToInput,
                fillSlider
            };

            fillSlider(fromSlider, toSlider, '#C6C6C6', '#5932EA', toSlider);
            fillSlider(goalFromSlider, goalToSlider, '#C6C6C6', '#5932EA', goalToSlider);
            fillSlider(raisedFromSlider, raisedToSlider, '#C6C6C6', '#5932EA', raisedToSlider);
        }

        updateSubcategoryOptions() {
            const selectedCategories = window.selectedCategories || new Set(['All Categories']);
            const subcategoryMap = window.categorySubcategoryMap || {};
            const subcategoryOptionsContainer = document.getElementById('subcategoryOptionsContainer');
            const subcategoryBtn = document.getElementById('subcategoryFilterBtn');

            let availableSubcategories = new Set();
            let isAllCategoriesSelected = selectedCategories.has('All Categories');

            if (isAllCategoriesSelected || selectedCategories.size === 0) {
                 (subcategoryMap['All Categories'] || []).forEach(subcat => availableSubcategories.add(subcat));
            } else {
                 availableSubcategories.add('All Subcategories'); 
                 selectedCategories.forEach(cat => {
                     (subcategoryMap[cat] || []).forEach(subcat => {
                         if (subcat !== 'All Subcategories') { 
                            availableSubcategories.add(subcat);
                         }
                     });
                 });
            }

            const sortedSubcategories = Array.from(availableSubcategories);
            sortedSubcategories.sort((a, b) => {
                if (a === 'All Subcategories') return -1;
                if (b === 'All Subcategories') return 1;
                return a.localeCompare(b);
            });

            subcategoryOptionsContainer.innerHTML = sortedSubcategories.map(opt =>
                `<div class="subcategory-option" data-value="${opt}">${opt}</div>`
            ).join('');

            window.selectedSubcategories = new Set(['All Subcategories']);
            const allSubcatOption = subcategoryOptionsContainer.querySelector('.subcategory-option[data-value="All Subcategories"]');
            if (allSubcatOption) {
                allSubcatOption.classList.add('selected');
            }
             this.updateButtonText(window.selectedSubcategories, subcategoryBtn, 'All Subcategories'); 


            this.setupMultiSelect(
                subcategoryOptionsContainer.querySelectorAll('.subcategory-option'),
                window.selectedSubcategories,
                'All Subcategories',
                subcategoryBtn,
                false 
            );
        }
    }

    function onRender(event) {
        if (!window.rendered) {
            window.tableManager = new TableManager();
            window.rendered = true;

            const resizeObserver = new ResizeObserver(() => {
                if (window.tableManager) {
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