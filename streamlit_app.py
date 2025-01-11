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
                    <link href='https://fonts.googleapis.com/css?family=Poppins' rel='stylesheet'>
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

# Calculate additional columns first
df['Goal'] = df['data.goal'].astype(float) * df['data.usd_exchange_rate'].astype(float)
df['%Raised'] = (df['data.converted_pledged_amount'].astype(float) / df['Goal']) * 100

df = df[[ 
    'data.name', 
    'data.creator.name', 
    'data.converted_pledged_amount', 
    'data.urls.web.project', 
    'data.location.expanded_country', 
    'data.state',
    # Hidden columns
    'data.category.parent_name',
    'data.category.name',
    'data.created_at',
    'Goal',
    '%Raised'
]].rename(columns={ 
    'data.name': 'Project Name', 
    'data.creator.name': 'Creator', 
    'data.converted_pledged_amount': 'Pledged Amount', 
    'data.urls.web.project': 'Link', 
    'data.location.expanded_country': 'Country', 
    'data.state': 'State',
    # Hidden columns
    'data.category.parent_name': 'Category',
    'data.category.name': 'Subcategory',
    'data.created_at': 'Date',
})

# Add formatting for numeric columns
df['Goal'] = df['Goal'].apply(lambda x: f"${x:,.2f}")
df['Pledged Amount'] = df['Pledged Amount'].apply(lambda x: f"${float(x):,.2f}")
df['%Raised'] = df['%Raised'].apply(lambda x: f"{x:.1f}%")

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
    # Define visible columns
    visible_columns = ['Project Name', 'Creator', 'Pledged Amount', 'Link', 'Country', 'State']
    
    # Generate header for visible columns only
    header_html = ''.join(f'<th scope="col">{column}</th>' for column in visible_columns)
    
    # Generate table rows with all columns but hide some
    rows_html = ''
    for _, row in df.iterrows():
        visible_cells = ''.join(f'<td>{row[col]}</td>' for col in visible_columns)
        hidden_cells = ''.join(f'<td class="hidden-cell">{row[col]}</td>' 
                             for col in df.columns if col not in visible_columns)
        rows_html += f'<tr class="table-row">{visible_cells}{hidden_cells}</tr>'
    
    return header_html, rows_html

# Generate table HTML
header_html, rows_html = generate_table_html(df)

# After loading data and before generating table, prepare filter options
def get_filter_options(df):
    return {
        'categories': sorted(['All Categories'] + df['Category'].unique().tolist()),
        'subcategories': sorted(['All Subcategories'] + df['Subcategory'].unique().tolist()),
        'countries': sorted(['All Countries'] + df['Country'].unique().tolist()),
        'states': sorted(['All States'] + df['State'].str.extract(r'>([^<]+)<')[0].unique().tolist()),
        'pledged_ranges': ['All Amounts'] + [
            f"${i}-${j}" for i, j in [(1,99), (100,999), (1000,9999), 
            (10000,99999), (100000,999999), (1000000,float('inf'))]
        ],
        'goal_ranges': ['All Goals'] + [
            f"${i}-${j}" for i, j in [(1,99), (100,999), (1000,9999), 
            (10000,99999), (100000,999999), (1000000,float('inf'))]
        ],
        'raised_ranges': ['All Percentages'] + [
            f"{i}%-{j}%" for i, j in [(0,20), (21,40), (41,60), (61,80), (81,100)]
        ],
        'date_ranges': [
            'All Time',
            'Last Month',
            'Last 6 Months',
            'Last Year',
            'Last 5 Years',
            'Last 10 Years',
            'Last 20 Years'
        ]
    }

filter_options = get_filter_options(df)

# Update template to include filter controls
template = f"""
<div class="filter-wrapper">
    <button class="reset-button" id="resetFilters">
        <span>Back to Default</span>
    </button>
    <div class="filter-controls">
        <div class="filter-row">
            <span class="filter-label">Explore</span>
            <select id="categoryFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['categories'])}
            </select>
            <span>&</span>
            <select id="subcategoryFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['subcategories'])}
            </select>
            <span>Projects On</span>
            <select id="countryFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['countries'])}
            </select>
            <span>Sorted By</span>
            <select id="sortFilter" class="filter-select">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
            </select>
        </div>
        <div class="filter-row">
            <span class="filter-label">More Flexible, Dynamic Search</span>
            <select id="stateFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['states'])}
            </select>
            <select id="pledgedFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['pledged_ranges'])}
            </select>
            <select id="goalFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['goal_ranges'])}
            </select>
            <select id="raisedFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['raised_ranges'])}
            </select>
            <select id="dateFilter" class="filter-select">
                {' '.join(f'<option value="{opt}">{opt}</option>' for opt in filter_options['date_ranges'])}
            </select>
        </div>
    </div>
</div>
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
        <button id="prev-page" class="page-btn" aria-label="Previous page">&lt;</button>
        <div id="page-numbers" class="page-numbers"></div>
        <button id="next-page" class="page-btn" aria-label="Next page">&gt;</button>
    </div>
</div>
"""

# Add new CSS styles
css = """
<style>
    .table-controls { 
        position: sticky; 
        top: 0; 
        background: #ffffff; 
        z-index: 2; 
        padding: 0 10px; 
        border-bottom: 1px solid #eee; 
        height: 60px; 
        display: flex; 
        align-items: center; 
        justify-content: flex-end; 
        margin-bottom: 1rem; 
        border-radius: 20px; 
    }
    
    .table-container { 
        position: relative;
        flex: 1;
        padding: 20px; 
        background: #ffffff; 
        min-height: 400px;
        overflow-y: visible;
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
        overflow: hidden; 
        text-overflow: ellipsis; 
        white-space: nowrap; 
        font-family: 'Poppins'; 
        font-size: 14px; 
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
    
    .state-live, .state-submitted { 
        background: #E6F3FF; 
        color: #0066CC; 
        border-color: #0066CC; 
    }

    .table-wrapper { 
        position: relative;
        display: flex;
        flex-direction: column;
        max-width: 100%; 
        background: #ffffff; 
        border-radius: 20px; 
        overflow: visible;
        min-height: 600px;
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
        position: relative;
        width: 100%;
        background: #ffffff;
        border-radius: 20px;
        margin-bottom: 20px;
        min-height: 120px;
        z-index: 3;
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
    }

    .filter-label {
        font-family: 'Poppins';
        font-size: 14px;
        color: #333;
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
        position: absolute;
        left: -40px;
        top: 0;
        height: 100%;
        width: 40px;
        background: #5932EA;
        color: white;
        border: none;
        border-radius: 8px 0 0 8px;
        cursor: pointer;
        padding: 0;
    }

    .reset-button span {
        transform: rotate(-90deg);
        white-space: nowrap;
        display: block;
        font-family: 'Poppins';
        font-size: 12px;
        letter-spacing: 1px;
    }

    .reset-button:hover {
        background: #4a29bb;
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

    class TableManager {
        constructor() {
            this.searchInput = document.getElementById('table-search');
            this.allRows = Array.from(document.querySelectorAll('#data-table tbody tr'));
            this.visibleRows = this.allRows;
            this.currentPage = 1;
            this.pageSize = 10;
            this.initialize();
            this.setupFilters();
        }

        initialize() {
            this.setupSearch();
            this.setupPagination();
            this.updateTable();
        }

        setupSearch() {
            const debouncedSearch = debounce((searchTerm) => {
                if (!searchTerm) {
                    this.visibleRows = this.allRows;
                } else {
                    const pattern = createRegexPattern(searchTerm);
                    this.visibleRows = this.allRows.filter(row => {
                        const text = row.textContent || row.innerText;
                        return pattern.test(text);
                    });
                }
                this.currentPage = 1;
                this.updateTable();
            }, 300);

            this.searchInput.addEventListener('input', (e) => {
                debouncedSearch(e.target.value.trim().toLowerCase());
            });
        }

        setupPagination() {
            document.getElementById('prev-page').onclick = () => this.previousPage();
            document.getElementById('next-page').onclick = () => this.nextPage();
            window.handlePageClick = (page) => this.goToPage(page);
        }

        updateTable() {
            // Hide all rows first
            this.allRows.forEach(row => row.style.display = 'none');
            
            // Show only visible rows for current page
            const start = (this.currentPage - 1) * this.pageSize;
            const end = start + this.pageSize;
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
            
            console.log(`Showing page ${this.currentPage} of ${totalPages}, ${this.visibleRows.length} total rows`);
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
                const filterWrapper = document.querySelector('.filter-wrapper');
                const tableWrapper = document.querySelector('.table-wrapper');
                const tableContainer = document.querySelector('.table-container');
                const table = document.querySelector('#data-table');
                const controls = document.querySelector('.table-controls');
                const pagination = document.querySelector('.pagination-controls');
                
                if (filterWrapper && tableWrapper && tableContainer && table) {
                    const filterHeight = filterWrapper.offsetHeight;
                    const tableHeight = table.offsetHeight;
                    const controlsHeight = controls.offsetHeight;
                    const paginationHeight = pagination.offsetHeight;
                    const padding = 40;
                    
                    // Calculate content height
                    const contentHeight = tableHeight + controlsHeight + paginationHeight + padding;
                    
                    // Calculate total component height
                    const totalHeight = filterHeight + contentHeight + padding;
                    
                    // Set minimum heights
                    const minContentHeight = 600; // Minimum height for table content
                    const finalHeight = Math.max(totalHeight, minContentHeight);
                    
                    // Update container heights
                    tableContainer.style.minHeight = `${Math.max(tableHeight, 400)}px`;
                    tableWrapper.style.minHeight = `${Math.max(contentHeight, minContentHeight)}px`;
                    
                    // Set final component height with additional padding
                    Streamlit.setFrameHeight(finalHeight + 40);
                    
                    console.log({
                        filterHeight,
                        tableHeight,
                        controlsHeight,
                        paginationHeight,
                        contentHeight,
                        finalHeight
                    });
                }
            });
        }

        setupFilters() {
            const filterIds = [
                'categoryFilter', 'subcategoryFilter', 'countryFilter', 'stateFilter',
                'pledgedFilter', 'goalFilter', 'raisedFilter', 'dateFilter', 'sortFilter'
            ];
            
            filterIds.forEach(id => {
                document.getElementById(id).addEventListener('change', () => this.applyFilters());
            });

            document.getElementById('resetFilters').addEventListener('click', () => this.resetFilters());
        }

        resetFilters() {
            const selects = document.querySelectorAll('.filter-select');
            selects.forEach(select => select.selectedIndex = 0);
            this.applyFilters();
        }

        applyFilters() {
            const filters = {
                category: document.getElementById('categoryFilter').value,
                subcategory: document.getElementById('subcategoryFilter').value,
                country: document.getElementById('countryFilter').value,
                state: document.getElementById('stateFilter').value,
                pledged: document.getElementById('pledgedFilter').value,
                goal: document.getElementById('goalFilter').value,
                raised: document.getElementById('raisedFilter').value,
                date: document.getElementById('dateFilter').value,
                sort: document.getElementById('sortFilter').value
            };

            this.visibleRows = this.allRows.filter(row => {
                const cells = Array.from(row.cells);
                const rowData = {
                    category: cells[6].textContent,
                    subcategory: cells[7].textContent,
                    country: cells[4].textContent,
                    state: cells[5].textContent.toLowerCase(),
                    pledged: parseFloat(cells[2].textContent.replace(/[^0-9.-]+/g,"")),
                    goal: parseFloat(cells[9].textContent.replace(/[^0-9.-]+/g,"")),
                    raised: parseFloat(cells[10].textContent),
                    date: new Date(cells[8].textContent)
                };

                return this.matchesFilters(rowData, filters);
            });

            if (filters.sort !== 'none') {
                this.sortRows(filters.sort);
            }

            this.currentPage = 1;
            this.updateTable();
        }

        matchesFilters(rowData, filters) {
            if (filters.category !== 'All Categories' && rowData.category !== filters.category) return false;
            if (filters.subcategory !== 'All Subcategories' && rowData.subcategory !== filters.subcategory) return false;
            if (filters.country !== 'All Countries' && rowData.country !== filters.country) return false;
            if (filters.state !== 'All States' && !rowData.state.includes(filters.state.toLowerCase())) return false;

            // Handle range filters
            if (filters.pledged !== 'All Amounts') {
                const [min, max] = filters.pledged.split('-').map(v => parseFloat(v.replace(/[^0-9.-]+/g,"")));
                if (rowData.pledged < min || (max && rowData.pledged > max)) return false;
            }

            if (filters.goal !== 'All Goals') {
                const [min, max] = filters.goal.split('-').map(v => parseFloat(v.replace(/[^0-9.-]+/g,"")));
                if (rowData.goal < min || (max && rowData.goal > max)) return false;
            }

            if (filters.raised !== 'All Percentages') {
                const [min, max] = filters.raised.split('-').map(v => parseFloat(v));
                if (rowData.raised < min || rowData.raised > max) return false;
            }

            if (filters.date !== 'All Time') {
                const now = new Date();
                let compareDate = new Date();
                
                switch(filters.date) {
                    case 'Last Month': compareDate.setMonth(now.getMonth() - 1); break;
                    case 'Last 6 Months': compareDate.setMonth(now.getMonth() - 6); break;
                    case 'Last Year': compareDate.setFullYear(now.getFullYear() - 1); break;
                    case 'Last 5 Years': compareDate.setFullYear(now.getFullYear() - 5); break;
                    case 'Last 10 Years': compareDate.setFullYear(now.getFullYear() - 10); break;
                    case 'Last 20 Years': compareDate.setFullYear(now.getFullYear() - 20); break;
                }
                
                if (rowData.date < compareDate) return false;
            }

            return true;
        }

        sortRows(sortType) {
            this.visibleRows.sort((a, b) => {
                const dateA = new Date(a.cells[8].textContent);
                const dateB = new Date(b.cells[8].textContent);
                return sortType === 'newest' ? dateB - dateA : dateA - dateB;
            });
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
table_component = gensimplecomponent('searchable_table', template=css + template, script=script)
table_component()
