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
    return f'<div class="state_cell state-{state}">{state}</div>'

# Apply styling to State column
df['State'] = df['State'].apply(style_state)

def generate_table_html(df):
    # Generate table header
    header_html = ''.join(f'<th>{column}</th>' for column in df.columns)
    
    # Generate table rows
    rows_html = ''
    for _, row in df.iterrows():
        cells = ''.join(f'<td>{value}</td>' for value in row)
        rows_html += f'<tr class="table-row">{cells}</tr>'
    
    return header_html, rows_html

# Generate table HTML
header_html, rows_html = generate_table_html(df)

# Create the complete HTML string
html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    .table-container {{ display: flex; justify-content: center; padding: 20px; }}
    table {{ border-collapse: collapse; width: 80%; max-width: 1200px; }}
    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
    td:has(.state_cell) {{ justify-items: center; }}
    .state_cell {{ width: 100%; padding: 3px 5px; text-align: center; border-radius: 4px; border: solid 1px; }}
    .state-canceled, .state-failed, .state-suspended {{ 
        background: #FFC5C5; color: #DF0404; border-color: #DF0404; 
    }}
    .state-successful {{ 
        background: #16C09861; color: #00B087; border-color: #00B087; 
    }}
    .state-live, .state-submitted {{ 
        background: #E6F3FF; color: #0066CC; border-color: #0066CC; 
    }}
    .table-wrapper {{ width: 100%; max-width: 1200px; margin: 0 auto; }}
    .table-controls {{ display: flex; justify-content: flex-end; margin-bottom: 1rem; padding: 0 10%; }}
    .search-input {{ 
        padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
        width: 200px; font-size: 14px; 
    }}
    .search-input:focus {{ 
        outline: none; border-color: #0066CC; 
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1); 
    }}
    .noscript-warning {{
        background-color: #fff3cd; color: #856404; padding: 12px;
        margin-bottom: 20px; border: 1px solid #ffeeba;
        border-radius: 4px; text-align: center; font-weight: 500;
    }}
</style>
</head>
<body>
    <div class="table-wrapper">
        <div class="table-controls">
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
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        function debounce(func, wait) {{
            let timeout;
            return function executedFunction(...args) {{
                const later = () => {{
                    clearTimeout(timeout);
                    func(...args);
                }};
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            }};
        }}

        function escapeRegex(string) {{
            return string.replace(/[.*+?^${{}}()|[\]\\]/g, '\\$&');
        }}

        function initializeSearch() {{
            console.log('Initializing search functionality...');
            const searchInput = document.getElementById('table-search');
            const tableRows = document.querySelectorAll('#data-table tbody tr');

            if (!searchInput || !tableRows.length) {{
                console.log('Elements not found, retrying...');
                setTimeout(initializeSearch, 100);
                return;
            }}

            const performSearch = (searchTerm) => {{
                try {{
                    const regex = new RegExp(escapeRegex(searchTerm), 'i');
                    let matchCount = 0;

                    tableRows.forEach(row => {{
                        const cells = Array.from(row.getElementsByTagName('td'));
                        const found = cells.some(cell => {{
                            const text = cell.textContent || cell.innerText;
                            return regex.test(text);
                        }});
                        
                        if (found) matchCount++;
                        row.style.display = found ? '' : 'none';
                    }});
                }} catch (e) {{
                    console.error('Search error:', e);
                }}
            }};

            const debouncedSearch = debounce((e) => {{
                const searchTerm = e.target.value.trim();
                performSearch(searchTerm);
            }}, 300);

            searchInput.addEventListener('input', debouncedSearch);
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeSearch);
        }} else {{
            initializeSearch();
        }}

        const observer = new MutationObserver(() => {{
            initializeSearch();
        }});

        observer.observe(document.body, {{
            childList: true,
            subtree: true
        }});
    </script>
</body>
</html>
"""

# Display the data
st.markdown(html_content, unsafe_allow_html=True)
