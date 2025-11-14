# 🚚 Simple Network Visualization

## Quick Start (3 Steps)

### 1. Install Dependencies
# pip install pgeocode
# pip install streamlit
# pip install pandas
# pip install plotly
# pip install networkx
# pip install numpy
# pip install pathlib
```bash
cd network_visualisation
python3 -m venv venv
source ./venv/bin/activate  
# On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 2. Verify Excel File
Make sure `Network_modelling_assignment_copy.xlsx` is in this folder. This is a restructured version of the original data.
### 3. Run Dashboard
```bash
streamlit run network_dashboard.py
```
The dashboard will open at: http://localhost:8501

## If You Get Errors
Clear Streamlit’s cache (either from the app menu → “Clear cache” or run streamlit cache clear in the repo root).

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip3 install -r requirements.txt
```

### "Excel file not found"
- Make sure the Excel file is in the same folder as `network_dashboard.py`
- Check the filename matches exactly: `Network_modelling_assignment_copy.xlsx`

### "No data found"
- Open the Excel file and verify it has a sheet named `ShipmentData`
- Check that the sheet has columns: `Origin State`, `DestinationState`, `Weight`, `ShipDate`

## What You'll See

The dashboard has 5 visualization types:

1. **Overview Dashboard** - Bar charts and summary stats
2. **Network Graph** - Interactive node-link diagram  
3. **Geographic Flow Map** - US map with flow lines
4. **Sankey Diagram** - Flow visualization
5. **Detailed Statistics** - Tables and analysis

Use the sidebar filters to explore data!
