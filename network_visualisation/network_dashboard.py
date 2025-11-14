import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import defaultdict
import numpy as np
from pathlib import Path
import pgeocode

# Paths and helpers for optimized scenarios
BASE_DIR = Path(__file__).resolve().parent
OPT_RESULTS_PATH = BASE_DIR.parent / 'SCM_Network2_Optimisation' / 'optimization_results_dock_compare.xlsx'

# Optional manual coordinate overrides (lat, lon) for key nodes
CUSTOM_NODE_COORDS = {
    # Suppliers
    'GA30043': {'latitude': 34.0031, 'longitude': -84.0126},  # Lawrenceville, GA
    'CA91720': {'latitude': 34.0630, 'longitude': -117.6020},  # Rancho Cucamonga, CA (ZIP 91720 not in DB)
    # X-Docks
    'NY10006': {'latitude': 40.7116, 'longitude': -74.0125},  # Battery Park NYC
    'GA30113': {'latitude': 33.9519, 'longitude': -83.3576},  # Athens, GA
    'TX75477': {'latitude': 33.6357, 'longitude': -96.6333},  # Princeton, TX
    'NC27695': {'latitude': 35.7866, 'longitude': -78.6815},  # Raleigh, NC
    'IL61849': {'latitude': 40.1164, 'longitude': -88.2434},  # Champaign, IL
}


def _extract_zip(code):
    """Return the 5-digit ZIP substring from alphanumeric location codes."""
    if pd.isna(code):
        return None
    digits = ''.join(ch for ch in str(code) if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(5)


@st.cache_resource
def get_zip_geocoder():
    """Instantiate and cache the ZIP geocoder."""
    return pgeocode.Nominatim('us')


@st.cache_data
def geocode_zips(zip_codes):
    """Retrieve latitude/longitude for a set of US ZIP codes."""
    clean_zips = sorted({_extract_zip(z) for z in zip_codes if _extract_zip(z)})
    if not clean_zips:
        return pd.DataFrame(columns=['postal_code', 'latitude', 'longitude'])

    nomi = get_zip_geocoder()
    geo_df = nomi.query_postal_code(clean_zips)
    if geo_df is None:
        return pd.DataFrame(columns=['postal_code', 'latitude', 'longitude'])

    geo_df = geo_df[['postal_code', 'latitude', 'longitude']].dropna()
    geo_df['postal_code'] = geo_df['postal_code'].astype(str).str.zfill(5)
    return geo_df


def attach_coordinates(nodes_df):
    """Return nodes with coordinates, using geocoder plus manual overrides."""
    if nodes_df.empty or 'Zip' not in nodes_df.columns:
        return nodes_df.assign(latitude=pd.NA, longitude=pd.NA)

    geo_df = geocode_zips(nodes_df['Zip'])
    nodes_geo = nodes_df.merge(
        geo_df,
        how='left',
        left_on='Zip',
        right_on='postal_code'
    )

    # Apply manual overrides where provided
    for node, coords in CUSTOM_NODE_COORDS.items():
        mask = nodes_geo['Node'] == node
        if mask.any():
            nodes_geo.loc[mask, 'latitude'] = coords['latitude']
            nodes_geo.loc[mask, 'longitude'] = coords['longitude']

    return nodes_geo


def _find_table_start(raw_df, marker, second_col=None):
    """Locate the header row index for a sub-table by marker text."""
    for idx, value in raw_df[0].items():
        if pd.isna(value):
            continue
        if str(value).strip() != marker:
            continue
        if second_col is None:
            return idx
        # Search for second_col across all columns in this row
        row_values = raw_df.iloc[idx].tolist()
        for col_val in row_values:
            if pd.notna(col_val) and str(col_val).strip() == second_col:
                return idx
    return None


def _extract_table(raw_df, header_idx):
    """Slice a sub-table starting at header_idx until the next blank row."""
    if header_idx is None or header_idx >= len(raw_df):
        return pd.DataFrame()

    header_row = raw_df.iloc[header_idx].tolist()
    # Determine number of valid columns (stop at first NaN)
    col_count = len([col for col in header_row if not pd.isna(col)])
    columns = [str(col) for col in header_row[:col_count]]

    data = []
    row_idx = header_idx + 1
    while row_idx < len(raw_df):
        row_vals = raw_df.iloc[row_idx].tolist()
        if all(pd.isna(v) for v in row_vals):
            break
        data.append(row_vals[:col_count])
        row_idx += 1

    df = pd.DataFrame(data, columns=columns)
    return df


@st.cache_data
def load_optimized_network(years=1, scenario_label='Exactly 1 X-Dock(s)', results_signature=None, results_file=None):
    """Parse optimized scenario output into edge and node records.
    
    Args:
        years: Time horizon
        scenario_label: Scenario name
        results_signature: Cache busting parameter
        results_file: Optional path to results file
    """
    # ``results_signature`` is unused besides cache busting; keep for Streamlit hashing
    _ = results_signature
    
    # Use provided results file or default
    results_path = results_file if results_file is not None else OPT_RESULTS_PATH

    if not results_path.exists():
        return None

    sheet_stub = scenario_label.replace(' ', '_')[:25]
    sheet_name = f"{years}Y_{sheet_stub}"

    try:
        raw_df = pd.read_excel(results_path, sheet_name=sheet_name, header=None)
    except ValueError:
        return None

    direct_header = _find_table_start(raw_df, 'Supplier', second_col='Customer')
    xdock_flow_header = _find_table_start(raw_df, 'Supplier', second_col='XDock')
    xdocks_header = _find_table_start(raw_df, 'XDock')

    direct_df = _extract_table(raw_df, direct_header)
    xdock_flow_df = _extract_table(raw_df, xdock_flow_header)
    xdocks_df = _extract_table(raw_df, xdocks_header)

    edge_rows = []
    supplier_nodes = set()
    customer_nodes = set()
    xdock_nodes = set()

    if not direct_df.empty:
        direct_df['Flow_kg'] = pd.to_numeric(direct_df['Flow_kg'], errors='coerce').fillna(0)
        for _, row in direct_df.iterrows():
            supplier_nodes.add(row['Supplier'])
            customer_nodes.add(row['Customer'])
            edge_rows.append({
                'Segment': 'Direct',
                'From_Node': row['Supplier'],
                'From_Zip': _extract_zip(row['Supplier']),
                'To_Node': row['Customer'],
                'To_Zip': _extract_zip(row['Customer']),
                'Flow_kg': row['Flow_kg']
            })

    if not xdock_flow_df.empty:
        xdock_flow_df['Flow_kg'] = pd.to_numeric(xdock_flow_df['Flow_kg'], errors='coerce').fillna(0)
        for _, row in xdock_flow_df.iterrows():
            supplier_nodes.add(row['Supplier'])
            customer_nodes.add(row['Customer'])
            xdock_nodes.add(row['XDock'])

            flow_val = row['Flow_kg']
            edge_rows.append({
                'Segment': 'Supplier→XDock',
                'From_Node': row['Supplier'],
                'From_Zip': _extract_zip(row['Supplier']),
                'To_Node': row['XDock'],
                'To_Zip': _extract_zip(row['XDock']),
                'Flow_kg': flow_val
            })
            edge_rows.append({
                'Segment': 'XDock→Customer',
                'From_Node': row['XDock'],
                'From_Zip': _extract_zip(row['XDock']),
                'To_Node': row['Customer'],
                'To_Zip': _extract_zip(row['Customer']),
                'Flow_kg': flow_val
            })

    if not xdocks_df.empty and 'XDock' in xdocks_df.columns:
        for xdock in xdocks_df['XDock'].dropna().unique():
            xdock_nodes.add(xdock)

    edges_df = pd.DataFrame(edge_rows)

    node_records = []
    for supplier in supplier_nodes:
        node_records.append({'Node': supplier, 'Node_Type': 'Supplier', 'Zip': _extract_zip(supplier)})
    for xdock in xdock_nodes:
        node_records.append({'Node': xdock, 'Node_Type': 'X-Dock', 'Zip': _extract_zip(xdock)})
    for customer in customer_nodes:
        node_records.append({'Node': customer, 'Node_Type': 'Customer', 'Zip': _extract_zip(customer)})

    nodes_df = pd.DataFrame(node_records)

    nodes_geo = attach_coordinates(nodes_df)

    return {
        'sheet_name': sheet_name,
        'edges': edges_df,
        'nodes': nodes_df,
        'nodes_geo': nodes_geo
    }


@st.cache_data
def load_brownfield_network():
    """Load brownfield study results from CSV files and build network structure."""
    brownfield_dir = BASE_DIR.parent / 'network3_brownfield'
    
    # Check if results exist
    routing_path = brownfield_dir / 'routing_decisions.csv'
    xdock_path = brownfield_dir / 'crossdock_summary.csv'
    shipment_path = brownfield_dir / 'Brownfield_Study.xlsx'
    
    if not routing_path.exists() or not xdock_path.exists():
        return None
    
    # Load results
    routing_df = pd.read_csv(routing_path)
    xdock_df = pd.read_csv(xdock_path)
    
    # Load shipment data to get destination details
    shipments = pd.read_excel(shipment_path, sheet_name='ShipmentData')
    
    # Get opened crossdocks
    opened_xdocks = xdock_df[xdock_df['Decision'] == 'OPEN']['Crossdock'].tolist()
    
    # Build edge and node records
    edge_rows = []
    supplier_nodes = set()
    customer_nodes = set()
    xdock_nodes = set(opened_xdocks)
    
    # Process routing decisions
    for _, route in routing_df.iterrows():
        date = route['Date']
        origin = route['Origin']
        routing_type = route['Routing']
        
        # Get shipments for this date-origin
        day_shipments = shipments[
            (shipments['ShipDate'] == pd.to_datetime(date)) & 
            (shipments['Origin'] == origin)
        ]
        
        supplier_nodes.add(origin)
        
        if routing_type == 'DIRECT':
            # Direct shipments: origin -> each customer
            for _, ship in day_shipments.iterrows():
                customer = ship['Destination']
                weight = ship['Weight']
                customer_nodes.add(customer)
                
                edge_rows.append({
                    'Segment': 'Direct',
                    'From_Node': origin,
                    'From_Zip': _extract_zip(origin),
                    'To_Node': customer,
                    'To_Zip': _extract_zip(customer),
                    'Flow_kg': weight,
                    'Date': date
                })
        else:
            # Via crossdock: origin -> xdock -> each customer
            xdock = route['Crossdock']
            xdock_nodes.add(xdock)
            
            # Aggregate weight going to this crossdock
            total_weight = day_shipments['Weight'].sum()
            
            # Inbound leg: origin -> crossdock
            edge_rows.append({
                'Segment': 'Supplier→XDock',
                'From_Node': origin,
                'From_Zip': _extract_zip(origin),
                'To_Node': xdock,
                'To_Zip': _extract_zip(xdock),
                'Flow_kg': total_weight,
                'Date': date
            })
            
            # Outbound legs: crossdock -> each customer
            for _, ship in day_shipments.iterrows():
                customer = ship['Destination']
                weight = ship['Weight']
                customer_nodes.add(customer)
                
                edge_rows.append({
                    'Segment': 'XDock→Customer',
                    'From_Node': xdock,
                    'From_Zip': _extract_zip(xdock),
                    'To_Node': customer,
                    'To_Zip': _extract_zip(customer),
                    'Flow_kg': weight,
                    'Date': date
                })
    
    # Aggregate edges by route (sum across all dates)
    edges_df = pd.DataFrame(edge_rows)
    if not edges_df.empty:
        edges_df = edges_df.groupby(
            ['Segment', 'From_Node', 'From_Zip', 'To_Node', 'To_Zip'],
            as_index=False
        )['Flow_kg'].sum()
    
    # Create node records
    node_records = []
    for supplier in supplier_nodes:
        node_records.append({'Node': supplier, 'Node_Type': 'Supplier', 'Zip': _extract_zip(supplier)})
    for xdock in xdock_nodes:
        node_records.append({'Node': xdock, 'Node_Type': 'X-Dock', 'Zip': _extract_zip(xdock)})
    for customer in customer_nodes:
        node_records.append({'Node': customer, 'Node_Type': 'Customer', 'Zip': _extract_zip(customer)})
    
    nodes_df = pd.DataFrame(node_records)
    nodes_geo = attach_coordinates(nodes_df)
    
    return {
        'edges': edges_df,
        'nodes': nodes_df,
        'nodes_geo': nodes_geo,
        'opened_xdocks': opened_xdocks,
        'xdock_summary': xdock_df
    }


def render_as_is_geographic_map(filtered_df):
    """Render the historic as-is state-level network map."""

    state_coords = {
        'GA': {'lat': 32.1656, 'lon': -82.9001},
        'CA': {'lat': 36.7783, 'lon': -119.4179},
        'NY': {'lat': 42.1657, 'lon': -74.9481},
        'TX': {'lat': 31.9686, 'lon': -99.9018},
        'IN': {'lat': 40.2672, 'lon': -86.1349},
        'CT': {'lat': 41.6032, 'lon': -73.0877},
        'MO': {'lat': 37.9643, 'lon': -91.8318},
        'AL': {'lat': 32.3182, 'lon': -86.9023},
        'IL': {'lat': 40.6331, 'lon': -89.3985},
        'VA': {'lat': 37.4316, 'lon': -78.6569},
        'MI': {'lat': 44.3148, 'lon': -85.6024},
        'FL': {'lat': 27.6648, 'lon': -81.5158},
        'NC': {'lat': 35.7596, 'lon': -79.0193},
        'OH': {'lat': 40.4173, 'lon': -82.9071},
        'PA': {'lat': 41.2033, 'lon': -77.1945},
        'TN': {'lat': 35.5175, 'lon': -86.5804},
        'SC': {'lat': 33.8361, 'lon': -81.1637},
        'AR': {'lat': 35.2010, 'lon': -91.8319},
        'KY': {'lat': 37.8393, 'lon': -84.2700},
        'MS': {'lat': 32.3547, 'lon': -89.3985}
    }

    route_summary = filtered_df.groupby(['Origin State', 'DestinationState']).agg({
        'ShipmentID': 'count',
        'Weight': 'sum'
    }).reset_index()

    fig = go.Figure()
    max_w = route_summary['Weight'].max() if len(route_summary) else 0
    min_width, max_width = 2.0, 10.0
    missing_states = set()

    for _, row in route_summary.iterrows():
        origin = row['Origin State']
        dest = row['DestinationState']

        if origin in state_coords and dest in state_coords:
            w = float(row['Weight'])
            scale = 0.0 if max_w == 0 else (w / max_w) ** 0.5
            width_px = min_width + (max_width - min_width) * scale
            fig.add_trace(go.Scattergeo(
                lon=[state_coords[origin]['lon'], state_coords[dest]['lon']],
                lat=[state_coords[origin]['lat'], state_coords[dest]['lat']],
                mode='lines',
                line=dict(width=width_px, color='rgba(200, 30, 30, 0.75)'),
                hoverinfo='text',
                text=f"{origin} → {dest}<br>Shipments: {row['ShipmentID']:,}<br>Weight: {row['Weight']:,.0f} kg",
                showlegend=False
            ))
        else:
            if origin not in state_coords:
                missing_states.add(origin)
            if dest not in state_coords:
                missing_states.add(dest)

    for origin in filtered_df['Origin State'].unique():
        if origin in state_coords:
            total_weight = filtered_df[filtered_df['Origin State'] == origin]['Weight'].sum()
            fig.add_trace(go.Scattergeo(
                lon=[state_coords[origin]['lon']],
                lat=[state_coords[origin]['lat']],
                mode='markers+text',
                marker=dict(size=20, color='green', symbol='square'),
                text=origin,
                textposition='top center',
                hoverinfo='text',
                hovertext=f"Supplier: {origin}<br>Total Weight: {total_weight:,.0f} kg",
                name='Suppliers',
                showlegend=True
            ))

    for dest in filtered_df['DestinationState'].unique():
        if dest in state_coords and dest not in filtered_df['Origin State'].unique():
            total_weight = filtered_df[filtered_df['DestinationState'] == dest]['Weight'].sum()
            fig.add_trace(go.Scattergeo(
                lon=[state_coords[dest]['lon']],
                lat=[state_coords[dest]['lat']],
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle'),
                hoverinfo='text',
                hovertext=f"Customer: {dest}<br>Total Weight: {total_weight:,.0f} kg",
                name='Customers',
                showlegend=True
            ))

    if len(missing_states) > 0:
        st.info(
            f"Some states are missing coordinates and were skipped in flow lines: {', '.join(sorted(missing_states))}"
        )

    fig.update_layout(
        title='Geographic Distribution of Shipments',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
        ),
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def render_optimized_geographic_map(years=1, scenario_label='Exactly 1 X-Dock(s)', results_file=None):
    """Render the MILP scenario map with supplier→dock→customer flows.
    
    Args:
        years: Time horizon (1-5)
        scenario_label: Scenario name (e.g., 'Exactly 2 X-Dock(s)')
        results_file: Optional path to results file (defaults to OPT_RESULTS_PATH)
    """
    
    # Use provided results file or default
    results_path = results_file if results_file is not None else OPT_RESULTS_PATH
    
    results_signature = None
    if results_path.exists():
        try:
            results_signature = results_path.stat().st_mtime
        except OSError:
            results_signature = None

    scenario_data = load_optimized_network(
        years=years,
        scenario_label=scenario_label,
        results_signature=results_signature,
        results_file=results_path
    )
    if scenario_data is None or scenario_data['edges'].empty:
        st.warning(
            "Optimized scenario results not found. Run network2_optimize_Cy_dock_compare before visualising."
        )
        return

    edges_df = scenario_data['edges']
    nodes_geo = scenario_data['nodes_geo']

    if nodes_geo.empty:
        st.error("Unable to attach coordinates to scenario nodes.")
        return

    coord_map = nodes_geo.set_index('Node')[['latitude', 'longitude', 'Node_Type']].to_dict('index')

    fig = go.Figure()
    color_map = {
        'Direct': 'rgba(30, 144, 255, 0.7)',
        'Supplier→XDock': 'rgba(34, 139, 34, 0.75)',
        'XDock→Customer': 'rgba(255, 140, 0, 0.75)'
    }

    max_flow = edges_df['Flow_kg'].max() if len(edges_df) else 0
    min_width, max_width = 2.0, 10.0
    missing_nodes = set()

    for _, edge in edges_df.iterrows():
        from_node = edge['From_Node']
        to_node = edge['To_Node']
        flow_val = float(edge['Flow_kg'])
        if flow_val <= 0:
            continue

        from_coords = coord_map.get(from_node)
        to_coords = coord_map.get(to_node)
        if not from_coords or not to_coords:
            if not from_coords:
                missing_nodes.add(from_node)
            if not to_coords:
                missing_nodes.add(to_node)
            continue

        scale = 0.0 if max_flow == 0 else (flow_val / max_flow) ** 0.5
        width_px = min_width + (max_width - min_width) * scale
        segment = edge.get('Segment', 'Direct')
        color = color_map.get(segment, 'rgba(128, 128, 128, 0.6)')

        fig.add_trace(go.Scattergeo(
            lon=[from_coords['longitude'], to_coords['longitude']],
            lat=[from_coords['latitude'], to_coords['latitude']],
            mode='lines',
            line=dict(width=width_px, color=color),
            hoverinfo='text',
            text=(
                f"{from_node} → {to_node}<br>Segment: {segment}<br>Weight: {flow_val:,.0f} kg"
            ),
            showlegend=False
        ))

    marker_styles = {
        'Supplier': dict(size=18, color='green', symbol='square', name='Suppliers', show_text=True),
        'X-Dock': dict(size=22, color='purple', symbol='star', name='X-Dock', show_text=True),
        'Customer': dict(size=12, color='red', symbol='circle', name='Customers', show_text=False)
    }

    for node_type, marker_style in marker_styles.items():
        subset = nodes_geo[nodes_geo['Node_Type'] == node_type].dropna(subset=['latitude', 'longitude'])
        if subset.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=subset['longitude'],
            lat=subset['latitude'],
            mode='markers+text' if marker_style['show_text'] else 'markers',
            marker=dict(
                size=marker_style['size'],
                color=marker_style['color'],
                symbol=marker_style['symbol']
            ),
            text=subset['Node'] if marker_style['show_text'] else None,
            textposition='top center',
            hoverinfo='text',
            hovertext=[
                f"{node_type}: {node}<br>ZIP: {zip_code}"
                for node, zip_code in zip(subset['Node'], subset['Zip'])
            ],
            name=marker_style['name'],
            showlegend=True
        ))

    if len(missing_nodes) > 0:
        st.info(
            "Some nodes were missing geocoded coordinates and were skipped: "
            + ', '.join(sorted(missing_nodes))
        )

    # Get X-Dock names for title
    xdock_names = nodes_geo[nodes_geo['Node_Type'] == 'X-Dock']['Node'].tolist()
    xdock_str = ', '.join(xdock_names) if xdock_names else 'No X-Docks'
    
    fig.update_layout(
        title=f'Optimized Network: {scenario_label} ({xdock_str})',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
        ),
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Display flow statistics
    st.markdown("### Network Flow Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Suppliers", len(nodes_geo[nodes_geo['Node_Type'] == 'Supplier']))
        suppliers = nodes_geo[nodes_geo['Node_Type'] == 'Supplier']['Node'].tolist()
        for sup in suppliers:
            st.write(f"  • {sup}")
    
    with col2:
        st.metric("X-Docks", len(nodes_geo[nodes_geo['Node_Type'] == 'X-Dock']))
        xdocks = nodes_geo[nodes_geo['Node_Type'] == 'X-Dock']['Node'].tolist()
        for xd in xdocks:
            st.write(f"  ⭐ {xd}")
    
    with col3:
        st.metric("Customers", len(nodes_geo[nodes_geo['Node_Type'] == 'Customer']))
        total_flow = edges_df['Flow_kg'].sum()
        st.write(f"Total Flow: {total_flow:,.0f} kg")
    
    # Flow breakdown
    st.markdown("### Flow Breakdown")
    supplier_to_xdock = edges_df[edges_df['Segment'] == 'Supplier→XDock']['Flow_kg'].sum()
    xdock_to_customer = edges_df[edges_df['Segment'] == 'XDock→Customer']['Flow_kg'].sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Supplier → X-Dock", f"{supplier_to_xdock:,.0f} kg", 
                  help="Green lines on map")
    with col2:
        st.metric("X-Dock → Customer", f"{xdock_to_customer:,.0f} kg",
                  help="Orange lines on map")
    
    st.caption("💡 Line thickness represents shipment weight. Hover over lines and nodes for details.")


def render_brownfield_geographic_map():
    """Render the brownfield study network with optimized routing decisions."""
    
    brownfield_data = load_brownfield_network()
    
    if brownfield_data is None:
        st.error(
            "❌ Brownfield results not found. Please run the optimization first:\n\n"
            "`cd network3_brownfield && python3 brownfield_study.py`"
        )
        return
    
    edges_df = brownfield_data['edges']
    nodes_geo = brownfield_data['nodes_geo']
    opened_xdocks = brownfield_data['opened_xdocks']
    xdock_summary = brownfield_data['xdock_summary']
    
    if nodes_geo.empty:
        st.error("Unable to attach coordinates to network nodes.")
        return
    
    coord_map = nodes_geo.set_index('Node')[['latitude', 'longitude', 'Node_Type']].to_dict('index')
    
    fig = go.Figure()
    color_map = {
        'Direct': 'rgba(30, 144, 255, 0.7)',
        'Supplier→XDock': 'rgba(34, 139, 34, 0.75)',
        'XDock→Customer': 'rgba(255, 140, 0, 0.75)'
    }
    
    max_flow = edges_df['Flow_kg'].max() if len(edges_df) else 0
    min_width, max_width = 2.0, 10.0
    missing_nodes = set()
    
    # Draw edges
    for _, edge in edges_df.iterrows():
        from_node = edge['From_Node']
        to_node = edge['To_Node']
        flow_val = float(edge['Flow_kg'])
        if flow_val <= 0:
            continue
        
        from_coords = coord_map.get(from_node)
        to_coords = coord_map.get(to_node)
        if not from_coords or not to_coords:
            if not from_coords:
                missing_nodes.add(from_node)
            if not to_coords:
                missing_nodes.add(to_node)
            continue
        
        scale = 0.0 if max_flow == 0 else (flow_val / max_flow) ** 0.5
        width_px = min_width + (max_width - min_width) * scale
        segment = edge.get('Segment', 'Direct')
        color = color_map.get(segment, 'rgba(128, 128, 128, 0.6)')
        
        fig.add_trace(go.Scattergeo(
            lon=[from_coords['longitude'], to_coords['longitude']],
            lat=[from_coords['latitude'], to_coords['latitude']],
            mode='lines',
            line=dict(width=width_px, color=color),
            hoverinfo='text',
            text=(
                f"{from_node} → {to_node}<br>Segment: {segment}<br>Weight: {flow_val:,.0f} kg"
            ),
            showlegend=False
        ))
    
    # Draw nodes
    marker_styles = {
        'Supplier': dict(size=18, color='green', symbol='square', name='Suppliers', show_text=True),
        'X-Dock': dict(size=22, color='purple', symbol='star', name='X-Docks (Opened)', show_text=True),
        'Customer': dict(size=12, color='red', symbol='circle', name='Customers', show_text=False)
    }
    
    for node_type, marker_style in marker_styles.items():
        subset = nodes_geo[nodes_geo['Node_Type'] == node_type].dropna(subset=['latitude', 'longitude'])
        if subset.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=subset['longitude'],
            lat=subset['latitude'],
            mode='markers+text' if marker_style['show_text'] else 'markers',
            marker=dict(
                size=marker_style['size'],
                color=marker_style['color'],
                symbol=marker_style['symbol']
            ),
            text=subset['Node'] if marker_style['show_text'] else None,
            textposition='top center',
            hoverinfo='text',
            hovertext=[
                f"{node_type}: {node}<br>ZIP: {zip_code}"
                for node, zip_code in zip(subset['Node'], subset['Zip'])
            ],
            name=marker_style['name'],
            showlegend=True
        ))
    
    if len(missing_nodes) > 0:
        st.info(
            "Some nodes were missing geocoded coordinates and were skipped: "
            + ', '.join(sorted(missing_nodes))
        )
    
    xdock_str = ', '.join(opened_xdocks) if opened_xdocks else 'No X-Docks'
    
    fig.update_layout(
        title=f'Brownfield Study: Optimized Network ({xdock_str})',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
        ),
        height=700,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display flow statistics
    st.markdown("### Network Flow Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Suppliers", len(nodes_geo[nodes_geo['Node_Type'] == 'Supplier']))
        suppliers = nodes_geo[nodes_geo['Node_Type'] == 'Supplier']['Node'].tolist()
        for sup in suppliers:
            st.write(f"  • {sup}")
    
    with col2:
        st.metric("X-Docks (Opened)", len(opened_xdocks))
        for xd in opened_xdocks:
            xd_info = xdock_summary[xdock_summary['Crossdock'] == xd].iloc[0]
            st.write(f"  ⭐ {xd}")
            st.caption(f"    Routes: {xd_info['Routes_Using']}, Weight: {xd_info['Total_Weight_kg']:,.0f} kg")
    
    with col3:
        st.metric("Customers", len(nodes_geo[nodes_geo['Node_Type'] == 'Customer']))
        total_flow = edges_df['Flow_kg'].sum()
        st.write(f"Total Flow: {total_flow:,.0f} kg")
    
    # Flow breakdown
    st.markdown("### Flow Breakdown")
    direct_flow = edges_df[edges_df['Segment'] == 'Direct']['Flow_kg'].sum()
    supplier_to_xdock = edges_df[edges_df['Segment'] == 'Supplier→XDock']['Flow_kg'].sum()
    xdock_to_customer = edges_df[edges_df['Segment'] == 'XDock→Customer']['Flow_kg'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Direct Shipping", f"{direct_flow:,.0f} kg", 
                  help="Blue lines on map - Direct from supplier to customer")
    with col2:
        st.metric("Supplier → X-Dock", f"{supplier_to_xdock:,.0f} kg", 
                  help="Green lines on map - Consolidated shipments to crossdock")
    with col3:
        st.metric("X-Dock → Customer", f"{xdock_to_customer:,.0f} kg",
                  help="Orange lines on map - Distribution from crossdock")
    
    # Cost savings display
    exec_summary_path = BASE_DIR.parent / 'network3_brownfield' / 'executive_summary.csv'
    if exec_summary_path.exists():
        exec_df = pd.read_csv(exec_summary_path)
        st.markdown("### Cost Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Cost (All Direct)", 
                     f"${exec_df['Baseline_Cost'].iloc[0]:,.0f}")
        with col2:
            st.metric("Optimized Cost", 
                     f"${exec_df['Optimized_Cost'].iloc[0]:,.0f}")
        with col3:
            st.metric("Total Savings", 
                     f"${exec_df['Total_Savings'].iloc[0]:,.0f}",
                     delta=f"{exec_df['Savings_Percent'].iloc[0]:.1f}%")
    
    st.caption("💡 Line thickness represents shipment weight. Hover over lines and nodes for details.")


# Page configuration
st.set_page_config(
    page_title="Supply Chain Network Analysis",
    page_icon="🚚",
    layout="wide"
)

# Load data
@st.cache_data # a streamlit decorator, wraps function so its return value is cached across reruns.
def load_data():
    # Load Excel file from same directory
    df = pd.read_excel('Network_modelling_assignment_copy.xlsx', 
                       sheet_name='ShipmentData')
    
    # Set first row as column names and remove it from data
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    
    # Convert weight to numeric and drop nulls
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['ShipDate'] = pd.to_datetime(df['ShipDate'], errors='coerce')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['Weight', 'ShipDate', 'Origin State', 'DestinationState'])
    
    return df

# Load data with error handling
try:
    df = load_data()
    # Display first few rows of the dataframe
    st.write("### First few rows of the data")
    st.dataframe(df.head())
    
    if len(df) == 0:
        st.error("No data found in Excel file. Please check the file.")
        st.stop()
except FileNotFoundError:
    st.error("❌ Excel file not found! Make sure 'Network_modelling_assignment_copy.xlsx' is in the same folder as this script.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Title and introduction
st.title("🚚 Supply Chain Network Analysis Dashboard")
st.markdown("### As-Is Network Visualization")
st.markdown("---")

# Sidebar for filters and controls
st.sidebar.header("Filters & Controls")

# Date range filter with null handling
min_date = df['ShipDate'].min()
max_date = df['ShipDate'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Origin filter
origin_filter = st.sidebar.multiselect(
    "Select Origin States",
    options=df['Origin State'].unique(),
    default=df['Origin State'].unique()
)

# Destination filter
dest_filter = st.sidebar.multiselect(
    "Select Destination States",
    options=df['DestinationState'].unique(),
    default=df['DestinationState'].unique()
)

# Weight range filter with null handling
weight_min = int(df['Weight'].min())
weight_max = int(df['Weight'].max())
min_weight, max_weight = st.sidebar.slider(
    "Shipment Weight Range (kg)",
    min_value=weight_min,
    max_value=weight_max,
    value=(weight_min, weight_max)
)

# Apply filters
filtered_df = df[
    (df['ShipDate'].dt.date >= date_range[0]) & # side bar date range start date
    (df['ShipDate'].dt.date <= date_range[1]) & # side bar date range end date
    (df['Origin State'].isin(origin_filter)) &
    (df['DestinationState'].isin(dest_filter)) &
    (df['Weight'] >= min_weight) &
    (df['Weight'] <= max_weight)
]

# Check if filtered data is empty
if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust her selections.")
    st.stop()

# Visualization type selector
st.sidebar.markdown("---")
st.sidebar.header("Visualization Type")
viz_type = st.sidebar.radio( # Function: Creates a set of mutually exclusive options, on sidebar
    "Choose visualization:",
    ["Overview Dashboard", "Network Graph", "Geographic Flow Map", "Sankey Diagram", "Detailed Statistics"]
)

# Calculate key metrics
total_shipments = len(filtered_df)
total_weight = filtered_df['Weight'].sum()
avg_weight = filtered_df['Weight'].mean()
unique_routes = filtered_df.groupby(['Origin State', 'DestinationState']).size().shape[0] 
# .size - A pandas method that returns the count of elements in each group
# .shape - A pandas/numpy attribute that returns a tuple representing the dimensions of the data

# Display key metrics at the top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Shipments", f"{total_shipments:,}")
with col2:
    st.metric("Total Weight", f"{total_weight:,.0f} kg")
with col3:
    st.metric("Avg Shipment Weight", f"{avg_weight:,.0f} kg")
with col4:
    st.metric("Unique Routes", unique_routes)

st.markdown("---")

# OVERVIEW DASHBOARD all toggles
if viz_type == "Overview Dashboard":
    st.header("📊 Network Overview")
    
    col1, col2 = st.columns(2) # creates 2 equal-width columns and returns them as a list
    
    with col1:
        # Shipments by origin
        origin_summary = filtered_df.groupby('Origin State').agg({
            'ShipmentID': 'count',
            'Weight': 'sum'
        }).reset_index()
        origin_summary.columns = ['Origin State', 'Shipment Count', 'Total Weight']
        
        fig = px.bar(origin_summary, x='Origin State', y='Shipment Count',
                     title='Shipments by Origin',
                     color='Total Weight',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top destinations
        dest_summary = filtered_df.groupby('DestinationState').agg({
            'ShipmentID': 'count',
            'Weight': 'sum'
        }).reset_index().sort_values('ShipmentID', ascending=False).head(10)
        dest_summary.columns = ['Destination State', 'Shipment Count', 'Total Weight']
        
        fig = px.bar(dest_summary, x='Destination State', y='Shipment Count',
                     title='Top 10 Destination States',
                     color='Total Weight',
                     color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Weight distribution
        fig = px.histogram(filtered_df, x='Weight', nbins=50,
                          title='Shipment Weight Distribution',
                          labels={'Weight': 'Weight (kg)', 'count': 'Number of Shipments'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Shipments over time
        daily_shipments = filtered_df.groupby(filtered_df['ShipDate'].dt.date).size().reset_index()
        daily_shipments.columns = ['Date', 'Shipments']
        
        fig = px.line(daily_shipments, x='Date', y='Shipments',
                     title='Shipments Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# NETWORK GRAPH
elif viz_type == "Network Graph":
    st.header("🕸️ Network Graph Visualization")
    
    # Aggregate shipments by route
    route_summary = filtered_df.groupby(['Origin State', 'DestinationState']).agg({
        'ShipmentID': 'count',
        'Weight': 'sum'
    }).reset_index()
    route_summary.columns = ['Origin', 'Destination', 'Shipment Count', 'Total Weight']
    
    # Create network graph
    G = nx.DiGraph()
    """Nodes (circles) represent locations
    Arrows (edges) show the direction of shipments
    Edge weights represent shipment volumes/weights """
    # Add nodes - handle states that are both origins and destinations
    for origin in route_summary['Origin'].unique():
        G.add_node(origin, 
                node_type='supplier',
                size=route_summary[route_summary['Origin'] == origin]['Total Weight'].sum())

    # Add all destinations from the data
    for dest in route_summary['Destination'].unique():
        if G.has_node(dest):
            # If node exists (is an origin), mark as both
            G.nodes[dest]['node_type'] = 'both'
        else:
            # Add as customer
            G.add_node(dest, 
                    node_type='customer',
                    size=route_summary[route_summary['Destination'] == dest]['Total Weight'].sum())
    
    # Add edges
    # Compute scaling for edge widths
    max_w = route_summary['Total Weight'].max()
    min_width, max_width = 2.0, 10.0  # px

    # Add edges with proper styling
    for _, row in route_summary.iterrows():
        w = float(row['Total Weight'])
        # sqrt scaling for better contrast; clamp when max_w == 0
        scale = 0.0 if max_w == 0 else (w / max_w) ** 0.5
        width_px = min_width + (max_width - min_width) * scale
        G.add_edge(
            row['Origin'],
            row['Destination'],
            weight=row['Total Weight'],
            shipments=row['Shipment Count'],
            width=width_px,
            color='rgba(60, 60, 60, 0.8)'
        )
    # Create arrows(edges)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge trace
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        shipments = G.edges[edge]['shipments']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
            width=G.edges[edge].get('width', 1),  # Use the width we set
            color=G.edges[edge].get('color', '#888')  # Use the color we set
        ),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}<br>Shipments: {shipments:,}<br>Weight: {weight:,.0f} kg",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node].get('node_type', 'customer')  # Default to customer if not set
        node_weight = G.nodes[node].get('size', 0)  # Default to 0 if not set
        
        if node_type == 'supplier':
            node_color.append('green')
            node_text.append(f"Supplier: {node}<br>Total Weight: {node_weight:,.0f} kg")
            node_size.append(40)
        elif node_type == 'both':
            node_color.append('blue')
            node_text.append(f"Supplier & Customer: {node}<br>Total Weight: {node_weight:,.0f} kg")
            node_size.append(30)
        else:
            node_color.append('red')
            node_text.append(f"Customer: {node}<br>Total Weight: {node_weight:,.0f} kg")
            node_size.append(20)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='Supply Chain Network Graph',
        showlegend=False,
        hovermode='closest',
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add legend
    st.markdown("""
    **Legend:**
    - 🟢 Green nodes = Suppliers
    - 🔴 Red nodes = Customers
    - Line thickness = Total weight shipped on route
    """)

# GEOGRAPHIC FLOW MAP
elif viz_type == "Geographic Flow Map":
    st.header("🗺️ Geographic Flow Map")

    geo_mode = st.sidebar.radio(
        "Select scenario:",
        [
            "As-Is Network (Direct Shipping)", 
            "Network 2: Optimized (2 X-Docks)",
            "Network 3: Brownfield Study"
        ]
    )

    if geo_mode == "As-Is Network (Direct Shipping)":
        st.subheader("Historical As-Is Network - Direct Shipping")
        st.info("This shows the historical shipment patterns with direct shipping from suppliers to customers.")
        render_as_is_geographic_map(filtered_df)
    
    elif geo_mode == "Network 2: Optimized (2 X-Docks)":
        st.subheader("Network 2: Optimized Network - Year 1 with 2 X-Docks")
        st.info("Most cost-effective X-Dock scenario: TX75477 (Texas) + GA30113 (Georgia). All flows route through X-Docks.")
        st.warning("⚠️ Note: Uses simplified rate structure ($0.005/kg/km for all shipments)")
        render_optimized_geographic_map(years=1, scenario_label='Exactly 2 X-Dock(s)')
    
    else:  # Network 3: Brownfield
        st.subheader("Network 3: Brownfield Study - Optimized with FTL/LTL Rates")
        st.success("✅ Realistic FTL/LTL rates with consolidation benefits")
        st.info("Analysis determines optimal crossdock opening decisions and routing strategies based on actual transportation costs.")
        render_brownfield_geographic_map()

# SANKEY DIAGRAM
elif viz_type == "Sankey Diagram":
    st.header("📊 Sankey Flow Diagram")
    
    # Aggregate by route
    route_summary = filtered_df.groupby(['Origin State', 'DestinationState']).agg({
        'Weight': 'sum'
    }).reset_index()
    
    # Create node labels
    sources = route_summary['Origin State'].unique().tolist()
    targets = route_summary['DestinationState'].unique().tolist()
    all_nodes = sources + [t for t in targets if t not in sources]
    
    # Create node indices
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Create links
    source_indices = [node_dict[row['Origin State']] for _, row in route_summary.iterrows()]
    target_indices = [node_dict[row['DestinationState']] for _, row in route_summary.iterrows()]
    values = route_summary['Weight'].tolist()
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=all_nodes,
            color=['green' if node in sources else 'red' for node in all_nodes]
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color='rgba(0,0,255,0.2)'
        )
    )])
    
    fig.update_layout(
        title="Supply Chain Flow (Sankey Diagram)",
        font=dict(size=12),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Left side (Green) = Suppliers
    - Right side (Red) = Customers
    - Flow width = Total weight shipped
    """)

# DETAILED STATISTICS
elif viz_type == "Detailed Statistics":
    st.header("📈 Detailed Network Statistics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Route Analysis", "Origin Analysis", "Destination Analysis", "Time Analysis"])
    
    with tab1:
        st.subheader("Route-Level Statistics")
        route_stats = filtered_df.groupby(['Origin State', 'DestinationState']).agg({
            'ShipmentID': 'count',
            'Weight': ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        route_stats.columns = ['Origin', 'Destination', 'Shipment Count', 'Total Weight', 
                               'Avg Weight', 'Std Weight', 'Min Weight', 'Max Weight']
        route_stats = route_stats.sort_values('Total Weight', ascending=False)
        
        st.dataframe(route_stats.style.format({
            'Total Weight': '{:,.0f}',
            'Avg Weight': '{:,.0f}',
            'Std Weight': '{:,.0f}',
            'Min Weight': '{:,.0f}',
            'Max Weight': '{:,.0f}'
        }), use_container_width=True)
        
        # Download button
        csv = route_stats.to_csv(index=False)
        st.download_button(
            label="Download Route Statistics as CSV",
            data=csv,
            file_name="route_statistics.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Origin (Supplier) Statistics")
        origin_stats = filtered_df.groupby('Origin State').agg({
            'ShipmentID': 'count',
            'Weight': 'sum',
            'DestinationState': 'nunique'
        }).reset_index()
        origin_stats.columns = ['Origin State', 'Total Shipments', 'Total Weight', 'Unique Destinations']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(origin_stats.style.format({
                'Total Weight': '{:,.0f}'
            }), use_container_width=True)
        
        with col2:
            fig = px.pie(origin_stats, values='Total Weight', names='Origin State',
                        title='Weight Distribution by Origin')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Destination (Customer) Statistics")
        dest_stats = filtered_df.groupby('DestinationState').agg({
            'ShipmentID': 'count',
            'Weight': ['sum', 'mean'],
            'Origin State': 'nunique'
        }).reset_index()
        dest_stats.columns = ['Destination State', 'Total Shipments', 'Total Weight', 
                             'Avg Weight per Shipment', 'Unique Origins']
        dest_stats = dest_stats.sort_values('Total Weight', ascending=False)
        
        st.dataframe(dest_stats.style.format({
            'Total Weight': '{:,.0f}',
            'Avg Weight per Shipment': '{:,.0f}'
        }), use_container_width=True)
        
        # Top 10 destinations chart
        fig = px.bar(dest_stats.head(10), x='Destination State', y='Total Shipments',
                    title='Top 10 Destinations by Shipment Count',
                    color='Total Weight',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Time-Based Analysis")
        
        # Monthly aggregation
        filtered_df['Month'] = filtered_df['ShipDate'].dt.to_period('M')
        monthly_stats = filtered_df.groupby('Month').agg({
            'ShipmentID': 'count',
            'Weight': 'sum'
        }).reset_index()
        monthly_stats['Month'] = monthly_stats['Month'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(monthly_stats, x='Month', y='ShipmentID',
                         title='Monthly Shipment Volume',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(monthly_stats, x='Month', y='Weight',
                         title='Monthly Weight Shipped',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📋 Data Summary")
st.write(f"**Total records in filtered data:** {len(filtered_df):,}")
st.write(f"**Date range:** {filtered_df['ShipDate'].min().date()} to {filtered_df['ShipDate'].max().date()}")
st.write(f"**Origins:** {', '.join(filtered_df['Origin State'].unique())}")
st.write(f"**Destinations:** {len(filtered_df['DestinationState'].unique())} states")
