import streamlit as st
import streamlit.components.v1 as components
import tempfile, os
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize

st.set_page_config(layout="wide")

def gensimplecomponent(name, template="", script=""):
    """Generate a simple Streamlit component."""
    def html():
        return f"""
            <!DOCTYPE html>
            <html lang="en">
                <head>
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

# Create table component template
template = f"""
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
    <div class="pagination-controls">
        <button id="prev-page" class="page-btn">&lt;</button>
        <span id="page-info">Page <span id="current-page">1</span> of <span id="total-pages">1</span></span>
        <button id="next-page" class="page-btn">&gt;</button>
        <select id="page-size" class="page-size-select">
            <option value="10">10 per page</option>
            <option value="20">20 per page</option>
            <option value="50">50 per page</option>
        </select>
    </div>
</div>
"""

# Create table component script with improved search
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
        rows.forEach(row => row.style.display = 'none');
        const start = (currentPage - 1) * pageSize;
        const end = Math.min(start + pageSize, rows.length);
        
        for (let i = start; i < end; i++) {
            if (rows[i]) rows[i].style.display = '';
        }
    }

    function updatePaginationControls(totalRows, currentPage, pageSize) {
        const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
        document.getElementById('total-pages').textContent = totalPages;
        document.getElementById('current-page').textContent = currentPage;
        document.getElementById('prev-page').disabled = currentPage <= 1;
        document.getElementById('next-page').disabled = currentPage >= totalPages;
    }

    class TablePagination {
        constructor(tableRows) {
            this.allRows = tableRows;
            this.visibleRows = tableRows;
            this.currentPage = 1;
            this.pageSize = 10;
            this.setupControls();
            this.updateTable();
        }

        setupControls() {
            const pageSizeSelect = document.getElementById('page-size');
            document.getElementById('prev-page').onclick = () => this.previousPage();
            document.getElementById('next-page').onclick = () => this.nextPage();
            pageSizeSelect.onchange = (e) => {
                this.pageSize = parseInt(e.target.value);
                this.currentPage = 1;
                this.updateTable();
            };
        }

        updateTable() {
            updateTableRows(this.visibleRows, this.currentPage, this.pageSize);
            updatePaginationControls(this.visibleRows.length, this.currentPage, this.pageSize);
            this.adjustHeight();
        }

        adjustHeight() {
            // Calculate total content height
            const wrapper = document.querySelector('.table-wrapper');
            const controls = document.querySelector('.table-controls');
            const table = document.querySelector('.table-container');
            const pagination = document.querySelector('.pagination-controls');
            
            if (wrapper && table) {
                // Add extra padding and ensure all elements are visible
                const totalHeight = controls.offsetHeight + 
                                  table.offsetHeight + 
                                  pagination.offsetHeight + 
                                  50; // extra padding
                
                Streamlit.setFrameHeight(totalHeight);
            }
        }
    }

    function initializeTable() {
        const searchInput = document.getElementById('table-search');
        const tableRows = Array.from(document.querySelectorAll('#data-table tbody tr'));
        
        if (!searchInput || !tableRows.length) {
            setTimeout(initializeTable, 100);
            return;
        }

        const pagination = new TablePagination(tableRows);
        const debouncedSearch = debounce(
            (term) => pagination.updateVisibleRows(term),
            300
        );

        searchInput.addEventListener('input', (e) => {
            debouncedSearch(e.target.value.trim().toLowerCase());
        });

        // Initial setup
        pagination.updateTable();
        
        // Add window resize handler
        window.addEventListener('resize', () => {
            pagination.adjustHeight();
        });
    }

    function onRender(event) {
        if (!window.rendered) {
            initializeTable();
            window.rendered = true;
        }
    }

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
"""

# Add CSS styles
css = """
<style>
    .table-container { 
        display: flex; 
        justify-content: center; 
        padding: 20px;
        width: 100%;
        background: #ffffff;
        min-height: 200px; /* Ensure minimum height */
    }
    table { 
        border-collapse: collapse; 
        width: 100%;
        background: #ffffff;
        table-layout: fixed;
    }
    th, td { 
        padding: 8px; 
        text-align: left; 
        border-bottom: 1px solid #ddd;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    th {
        background: #ffffff;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .state_cell { width: 100%; padding: 3px 5px; text-align: center; border-radius: 4px; border: solid 1px; }
    .state-canceled, .state-failed, .state-suspended { 
        background: #FFC5C5; color: #DF0404; border-color: #DF0404; 
    }
    .state-successful { 
        background: #16C09861; color: #00B087; border-color: #00B087; 
    }
    .state-live, .state-submitted { 
        background: #E6F3FF; color: #0066CC; border-color: #0066CC; 
    }
    .table-wrapper { 
        width: 100%; 
        background: #ffffff;
        border-radius: 20px;
        overflow: visible; /* Changed from auto to visible */
        display: flex;
        flex-direction: column;
    }
    .table-controls { display: flex; justify-content: flex-end; margin-bottom: 1rem; padding: 0 10%; }
    .search-input { 
        padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
        width: 200px; font-size: 14px; 
    }
    .search-input:focus { 
        outline: none; border-color: #0066CC; 
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1); 
    }
    .noscript-warning {
        background-color: #fff3cd; color: #856404; padding: 12px;
        margin-bottom: 20px; border: 1px solid #ffeeba;
        border-radius: 4px; text-align: center; font-weight: 500;
    }
    .pagination-controls {
        position: relative; /* Ensure pagination stays in flow */
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 1rem;
        gap: 0.5rem;
        background: #ffffff;
    }
    
    .page-btn {
        padding: 4px 8px;
        border: 1px solid #ddd;
        background: #fff;
        border-radius: 4px;
        cursor: pointer;
    }
    
    .page-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .page-size-select {
        padding: 4px 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-left: 1rem;
    }
    
    #page-info {
        margin: 0 0.5rem;
    }
</style>
"""

# Create and use the component
table_component = gensimplecomponent('searchable_table', template=css + template, script=script)
table_component()
