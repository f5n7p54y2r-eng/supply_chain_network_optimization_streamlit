import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import defaultdict
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Supply Chain Network Analysis",
    page_icon="🚚",
    layout="wide"
)


# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('data/Network_modelling_assignment_copy.xlsx',
                       sheet_name='ShipmentData', skiprows=0)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Convert weight to numeric
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['ShipDate'] = pd.to_datetime(df['ShipDate'], errors='coerce')

    return df


# Load data
df = load_data()

# Title and introduction
st.title("🚚 Supply Chain Network Analysis Dashboard")
st.markdown("### As-Is Network Visualization")
st.markdown("---")

# Sidebar for filters and controls
st.sidebar.header("Filters & Controls")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['ShipDate'].min(), df['ShipDate'].max()),
    min_value=df['ShipDate'].min().date(),
    max_value=df['ShipDate'].max().date()
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

# Weight range filter
min_weight, max_weight = st.sidebar.slider(
    "Shipment Weight Range (lbs)",
    min_value=int(df['Weight'].min()),
    max_value=int(df['Weight'].max()),
    value=(int(df['Weight'].min()), int(df['Weight'].max()))
)

# Apply filters
filtered_df = df[
    (df['ShipDate'].dt.date >= date_range[0]) &
    (df['ShipDate'].dt.date <= date_range[1]) &
    (df['Origin State'].isin(origin_filter)) &
    (df['DestinationState'].isin(dest_filter)) &
    (df['Weight'] >= min_weight) &
    (df['Weight'] <= max_weight)
    ]

# Visualization type selector
st.sidebar.markdown("---")
st.sidebar.header("Visualization Type")
viz_type = st.sidebar.radio(
    "Choose visualization:",
    ["Overview Dashboard", "Network Graph", "Geographic Flow Map", "Sankey Diagram", "Detailed Statistics"]
)

# Calculate key metrics
total_shipments = len(filtered_df)
total_weight = filtered_df['Weight'].sum()
avg_weight = filtered_df['Weight'].mean()
unique_routes = filtered_df.groupby(['Origin State', 'DestinationState']).size().shape[0]

# Display key metrics at the top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Shipments", f"{total_shipments:,}")
with col2:
    st.metric("Total Weight", f"{total_weight:,.0f} lbs")
with col3:
    st.metric("Avg Shipment Weight", f"{avg_weight:,.0f} lbs")
with col4:
    st.metric("Unique Routes", unique_routes)

st.markdown("---")

# OVERVIEW DASHBOARD
if viz_type == "Overview Dashboard":
    st.header("📊 Network Overview")

    col1, col2 = st.columns(2)

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
                           labels={'Weight': 'Weight (lbs)', 'count': 'Number of Shipments'})
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

    # Add nodes
    for origin in route_summary['Origin'].unique():
        G.add_node(origin, node_type='supplier',
                   size=route_summary[route_summary['Origin'] == origin]['Total Weight'].sum())

    for dest in route_summary['Destination'].unique():
        G.add_node(dest, node_type='customer',
                   size=route_summary[route_summary['Destination'] == dest]['Total Weight'].sum())

    # Add edges
    for _, row in route_summary.iterrows():
        G.add_edge(row['Origin'], row['Destination'],
                   weight=row['Total Weight'],
                   shipments=row['Shipment Count'])

    # Create layout
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
            line=dict(width=np.log1p(weight) / 1000, color='#888'),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}<br>Shipments: {shipments:,}<br>Weight: {weight:,.0f} lbs",
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

        node_type = G.nodes[node]['node_type']
        node_weight = G.nodes[node]['size']

        if node_type == 'supplier':
            node_color.append('green')
            node_text.append(f"Supplier: {node}<br>Total Weight: {node_weight:,.0f} lbs")
            node_size.append(40)
        else:
            node_color.append('red')
            node_text.append(f"Customer: {node}<br>Total Weight: {node_weight:,.0f} lbs")
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

    # State coordinates (simplified - using approximate center of each state)
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
        'SC': {'lat': 33.8361, 'lon': -81.1637}
    }

    # Aggregate by route
    route_summary = filtered_df.groupby(['Origin State', 'DestinationState']).agg({
        'ShipmentID': 'count',
        'Weight': 'sum'
    }).reset_index()

    # Create map
    fig = go.Figure()

    # Add flow lines
    for _, row in route_summary.iterrows():
        origin = row['Origin State']
        dest = row['DestinationState']

        if origin in state_coords and dest in state_coords:
            fig.add_trace(go.Scattergeo(
                lon=[state_coords[origin]['lon'], state_coords[dest]['lon']],
                lat=[state_coords[origin]['lat'], state_coords[dest]['lat']],
                mode='lines',
                line=dict(width=np.log1p(row['Weight']) / 500, color='rgba(255, 0, 0, 0.4)'),
                hoverinfo='text',
                text=f"{origin} → {dest}<br>Shipments: {row['ShipmentID']:,}<br>Weight: {row['Weight']:,.0f} lbs",
                showlegend=False
            ))

    # Add supplier nodes
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
                hovertext=f"Supplier: {origin}<br>Total Weight: {total_weight:,.0f} lbs",
                name='Suppliers',
                showlegend=True
            ))

    # Add customer nodes
    for dest in filtered_df['DestinationState'].unique():
        if dest in state_coords and dest not in filtered_df['Origin State'].unique():
            total_weight = filtered_df[filtered_df['DestinationState'] == dest]['Weight'].sum()
            fig.add_trace(go.Scattergeo(
                lon=[state_coords[dest]['lon']],
                lat=[state_coords[dest]['lat']],
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle'),
                hoverinfo='text',
                hovertext=f"Customer: {dest}<br>Total Weight: {total_weight:,.0f} lbs",
                name='Customers',
                showlegend=True
            ))

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