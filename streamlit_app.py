import streamlit as st
import streamlit.components.v1 as components
import tempfile, os
from pymongo import MongoClient
import pandas as pd
from pandas import json_normalize

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

# Create table component template (without CSS)
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
</div>
"""

# Create table component script with CSS injection
script = """
    function injectStyles(css) {
        const style = document.createElement('style');
        style.textContent = css;
        document.head.appendChild(style);
    }

    function onRender(event) {
        if (!window.rendered) {
            // Inject CSS
            const styles = `
                .table-container { display: flex; justify-content: center; padding: 20px; }
                table { border-collapse: collapse; width: 80%; max-width: 1200px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                td:has(.state_cell) { justify-items: center; }
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
                .table-wrapper { width: 100%; max-width: 1200px; margin: 0 auto; }
                .table-controls { display: flex; justify-content: flex-end; margin-bottom: 1rem; padding: 0 10%; }
                .search-input { 
                    padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
                    width: 200px; font-size: 14px; 
                }
                .search-input:focus { 
                    outline: none; border-color: #0066CC; 
                    box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1); 
                }
            `;
            injectStyles(styles);

            // Initialize search functionality
            const searchInput = document.getElementById('table-search');
            const tableRows = Array.from(document.querySelectorAll('#data-table tbody tr'));
            
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    const searchTerm = e.target.value.toLowerCase();
                    tableRows.forEach(row => {
                        const text = row.textContent.toLowerCase();
                        row.style.display = text.includes(searchTerm) ? '' : 'none';
                    });
                }, 300);
            });

            Streamlit.setFrameHeight(600);
            window.rendered = true;
        }
    }

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
"""

# Create and use the component (without CSS in template)
table_component = gensimplecomponent('searchable_table', template=template, script=script)
table_component()
