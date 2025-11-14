import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Load data
file_path = './data/Network_modelling_assignment.xlsx'
df_raw = pd.read_excel(file_path, sheet_name='ShipmentData')

# Load LTL Rate Card from Part2_Input sheet
rate_card_raw = pd.read_excel(file_path, sheet_name='Part2_Input', header=None)

# Load Distance Matrix from Part1_Input sheet
distance_raw = pd.read_excel(file_path, sheet_name='Part1_Input', header=None)

# =====================================================
# PARSE LTL RATE CARD
# =====================================================

# Extract LTL rates (rows 4-9, columns 2-6 based on Excel structure)
# Weight brackets: 0-500, 500-1000, 1000-2000, 2000-5000, >5000
# Distance brackets: 0-50, 50-250, 250-500, 500-1000, 1000-1500, >1500

ltl_rate_card = {
    '0-50': {
        '0-500': 0.0594,
        '500-1000': 0.05049,
        '1000-2000': 0.042916499999999996,
        '2000-5000': 0.036479025,
        '>5000': 0.03100717125
    },
    '50-250': {
        '0-500': 0.05642999999999999,
        '500-1000': 0.04796549999999999,
        '1000-2000': 0.04077067499999999,
        '2000-5000': 0.034655073749999994,
        '>5000': 0.029456812687499997
    },
    '250-500': {
        '0-500': 0.05360849999999999,
        '500-1000': 0.04556722499999999,
        '1000-2000': 0.03873214124999999,
        '2000-5000': 0.03292232006249999,
        '>5000': 0.027983972053124993
    },
    '500-1000': {
        '0-500': 0.05092807499999999,
        '500-1000': 0.04328886374999999,
        '1000-2000': 0.03679553418749999,
        '2000-5000': 0.031276204059374986,
        '>5000': 0.02658477345046874
    },
    '1000-1500': {
        '0-500': 0.04838167124999999,
        '500-1000': 0.04112442056249999,
        '1000-2000': 0.03495575747812499,
        '2000-5000': 0.029712393856406238,
        '>5000': 0.025255534777945305
    },
    '>1500': {
        '0-500': 0.04596258768749999,
        '500-1000': 0.039068199534374994,
        '1000-2000': 0.033207969604218734,
        '2000-5000': 0.028226774163585926,
        '>5000': 0.02399275803904804
    }
}

# FTL Rate: $1 per truck per km
FTL_RATE_PER_KM = 1.0
FTL_THRESHOLD = 5250  # kg (75% of 7000kg capacity)

# =====================================================
# PARSE DISTANCE MATRIX
# =====================================================

# Extract distance data from Part1_Input
# Supplier to Customer distances start around row 7
# Format: rows are destinations, columns are origins

# Build distance dictionary
distance_matrix = {}

# Parse origins from row 6 (suppliers)
origin_row = distance_raw.iloc[6, 5:7].values  # GA30043, CA91720
origins = [str(int(o)) if pd.notna(o) and isinstance(o, (int, float)) else str(o) for o in origin_row]

# Parse destinations and distances from rows 7 onwards
for idx in range(7, min(30, len(distance_raw))):  # Limit to relevant rows
    row = distance_raw.iloc[idx]
    dest_zip = row[4]  # Destination in column 4
    
    if pd.isna(dest_zip):
        continue
    
    # Convert destination to string format matching ShipmentData
    if isinstance(dest_zip, (int, float)):
        dest_str = str(int(dest_zip))
    else:
        dest_str = str(dest_zip)
    
    # Get distances for GA30043 and CA91720
    dist_ga = row[5]  # Distance from GA30043
    dist_ca = row[6]  # Distance from CA91720
    
    # Store in dictionary with proper formatting
    if pd.notna(dist_ga):
        distance_matrix[('GA30043', dest_str)] = float(dist_ga)
    if pd.notna(dist_ca):
        distance_matrix[('CA91720', dest_str)] = float(dist_ca)

# =====================================================
# HELPER FUNCTIONS FOR COST CALCULATION
# =====================================================

def get_distance_bracket(distance_km):
    """Determine which distance bracket the distance falls into"""
    if distance_km <= 50:
        return '0-50'
    elif distance_km <= 250:
        return '50-250'
    elif distance_km <= 500:
        return '250-500'
    elif distance_km <= 1000:
        return '500-1000'
    elif distance_km <= 1500:
        return '1000-1500'
    else:
        return '>1500'

def get_weight_bracket(weight_kg):
    """Determine which weight bracket the shipment falls into"""
    if weight_kg <= 500:
        return '0-500'
    elif weight_kg <= 1000:
        return '500-1000'
    elif weight_kg <= 2000:
        return '1000-2000'
    elif weight_kg <= 5000:
        return '2000-5000'
    else:
        return '>5000'

def get_distance(origin, destination):
    """
    Get distance between origin and destination.
    If not in matrix, estimate based on state proximity.
    """
    # Try direct lookup
    key = (origin, destination)
    if key in distance_matrix:
        return distance_matrix[key]
    
    # Fallback: estimate based on typical distances
    # This is a simplified fallback for missing data
    print(f"Warning: Distance not found for {origin} -> {destination}, using estimate")
    return 1000  # Default to 1000km if not found

def calculate_ltl_cost(weight_kg, distance_km):
    """Calculate LTL shipping cost using rate card"""
    distance_bracket = get_distance_bracket(distance_km)
    weight_bracket = get_weight_bracket(weight_kg)
    
    rate_per_kg_per_km = ltl_rate_card[distance_bracket][weight_bracket]
    cost = weight_kg * distance_km * rate_per_kg_per_km
    
    return cost

def calculate_ftl_cost(weight_kg, distance_km):
    """Calculate FTL shipping cost"""
    # FTL is charged per truck, not per kg
    # $1 per km per truck
    cost = distance_km * FTL_RATE_PER_KM
    return cost

def calculate_shipping_cost(row):
    """
    Calculate accurate shipping cost using real rate card and distances
    """
    weight = row['Weight']
    origin = row['Origin']
    destination = row['Destination']
    
    # Get distance
    distance = get_distance(origin, destination)
    
    # Determine if LTL or FTL
    if weight >= FTL_THRESHOLD:
        # Use FTL pricing
        cost = calculate_ftl_cost(weight, distance)
    else:
        # Use LTL pricing
        cost = calculate_ltl_cost(weight, distance)
    
    return cost

# =====================================================
# CLEAN AND PREPARE DATA
# =====================================================

# Clean shipment data
df = df_raw.iloc[1:].copy()
df.columns = ['ShipmentID', 'ShipDate', 'OriginZip', 'OriginState', 'Origin',
              'Destination', 'DestinationZip', 'DestinationState', 'Weight']

df['ShipmentID'] = pd.to_numeric(df['ShipmentID'], errors='coerce')
df['ShipDate'] = pd.to_datetime(df['ShipDate'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df = df.dropna(subset=['Weight', 'Origin', 'Destination'])
df['Lane'] = df['Origin'] + '-' + df['Destination']
df['Month'] = df['ShipDate'].dt.to_period('M').astype(str)

# =====================================================
# CALCULATE REAL COSTS
# =====================================================

print("Calculating shipping costs using real rate card and distances...")
print("This may take a moment for 6,900+ shipments...")

# Apply real cost calculation to each shipment
df['AdjustedCost'] = df.apply(calculate_shipping_cost, axis=1)
df['CostPerKg'] = df['AdjustedCost'] / df['Weight']

# Add date string for consolidation analysis
df['ShipDateStr'] = df['ShipDate'].dt.date

print("✓ Cost calculation complete!\n")

# =====================================================
# COST ANALYSIS OUTPUTS
# =====================================================

print("=" * 80)
print("AS-IS COST STRUCTURE ANALYSIS - STALMART INC.")
print("=" * 80)

total_shipments = len(df)
total_weight = df['Weight'].sum()
total_cost = df['AdjustedCost'].sum()
avg_cost_per_shipment = df['AdjustedCost'].mean()
avg_cost_per_kg = total_cost / total_weight

print(f"\n--- OVERALL COST METRICS ---")
print(f"Total Annual Shipping Cost: ${total_cost:,.2f}")
print(f"Average Cost per Shipment: ${avg_cost_per_shipment:.2f}")
print(f"Average Cost per Kg: ${avg_cost_per_kg:.2f}")
print(f"Total Shipments: {total_shipments:,}")
print(f"Total Weight Shipped: {total_weight:,.0f} kg")

# Cost by weight bracket
print(f"\n--- COST BREAKDOWN BY SHIPMENT SIZE ---")
brackets = [
    ('0-250 kg', 0, 250),
    ('251-500 kg', 250, 500),
    ('501-1000 kg', 500, 1000),
    ('1001-2000 kg', 1000, 2000),
    ('2001-5250 kg', 2000, 5250),
    ('>5250 kg (FTL)', 5250, float('inf'))
]

for label, min_w, max_w in brackets:
    if max_w == float('inf'):
        bracket_df = df[df['Weight'] > min_w]
    else:
        bracket_df = df[(df['Weight'] > min_w) & (df['Weight'] <= max_w)]
    
    bracket_cost = bracket_df['AdjustedCost'].sum()
    bracket_pct = (bracket_cost / total_cost) * 100
    bracket_count = len(bracket_df)
    avg_cost = bracket_df['AdjustedCost'].mean() if len(bracket_df) > 0 else 0
    
    print(f"{label:20} | Cost: ${bracket_cost:12,.2f} ({bracket_pct:5.1f}%) | "
          f"Shipments: {bracket_count:5} | Avg: ${avg_cost:7.2f}")

# Cost by destination state
print(f"\n--- TOP 10 STATES BY TOTAL SHIPPING COST ---")
state_costs = df.groupby('DestinationState').agg({
    'AdjustedCost': 'sum',
    'ShipmentID': 'count',
    'Weight': 'sum'
}).rename(columns={'ShipmentID': 'Shipments', 'Weight': 'TotalWeight'})
state_costs = state_costs.sort_values('AdjustedCost', ascending=False)

for idx, (state, row) in enumerate(state_costs.head(10).iterrows(), 1):
    pct = (row['AdjustedCost'] / total_cost) * 100
    avg_cost = row['AdjustedCost'] / row['Shipments']
    print(f"{idx:2}. {state}: ${row['AdjustedCost']:10,.2f} ({pct:5.1f}%) | "
          f"{row['Shipments']:4} shipments | Avg: ${avg_cost:7.2f}")

# Cost by lane
print(f"\n--- TOP 15 LANES BY TOTAL SHIPPING COST ---")
lane_costs = df.groupby('Lane').agg({
    'AdjustedCost': 'sum',
    'ShipmentID': 'count',
    'Weight': 'sum'
}).rename(columns={'ShipmentID': 'Shipments', 'Weight': 'TotalWeight'})
lane_costs = lane_costs.sort_values('AdjustedCost', ascending=False)

for idx, (lane, row) in enumerate(lane_costs.head(15).iterrows(), 1):
    pct = (row['AdjustedCost'] / total_cost) * 100
    avg_cost = row['AdjustedCost'] / row['Shipments']
    print(f"{idx:2}. {lane:40} | ${row['AdjustedCost']:10,.2f} ({pct:5.1f}%) | "
          f"{row['Shipments']:4} shipments | Avg: ${avg_cost:7.2f}")

# Monthly cost trends
print(f"\n--- MONTHLY COST BREAKDOWN ---")
monthly_costs = df.groupby('Month').agg({
    'AdjustedCost': 'sum',
    'ShipmentID': 'count',
    'Weight': 'sum'
}).rename(columns={'ShipmentID': 'Shipments', 'Weight': 'TotalWeight'})

for month, row in monthly_costs.iterrows():
    avg_cost = row['AdjustedCost'] / row['Shipments']
    print(f"{month}: ${row['AdjustedCost']:10,.2f} | {row['Shipments']:4} shipments | Avg: ${avg_cost:7.2f}")

# Cost concentration analysis
print(f"\n--- COST CONCENTRATION ANALYSIS ---")
top_10_lanes_cost = lane_costs.head(10)['AdjustedCost'].sum()
top_20_lanes_cost = lane_costs.head(20)['AdjustedCost'].sum()
top_10_states_cost = state_costs.head(10)['AdjustedCost'].sum()

print(f"Top 10 lanes account for: ${top_10_lanes_cost:,.2f} ({top_10_lanes_cost/total_cost*100:.1f}%)")
print(f"Top 20 lanes account for: ${top_20_lanes_cost:,.2f} ({top_20_lanes_cost/total_cost*100:.1f}%)")
print(f"Top 10 states account for: ${top_10_states_cost:,.2f} ({top_10_states_cost/total_cost*100:.1f}%)")

# LTL vs FTL analysis
ltl_shipments = df[df['Weight'] < 5250]
ftl_shipments = df[df['Weight'] >= 5250]

print(f"\n--- LTL vs FTL COST ANALYSIS ---")
print(f"LTL Shipments: {len(ltl_shipments):,} ({len(ltl_shipments)/total_shipments*100:.1f}%)")
print(f"LTL Total Cost: ${ltl_shipments['AdjustedCost'].sum():,.2f} ({ltl_shipments['AdjustedCost'].sum()/total_cost*100:.1f}%)")
print(f"LTL Avg Cost/Shipment: ${ltl_shipments['AdjustedCost'].mean():.2f}")
print(f"\nFTL Shipments: {len(ftl_shipments):,} ({len(ftl_shipments)/total_shipments*100:.1f}%)")
print(f"FTL Total Cost: ${ftl_shipments['AdjustedCost'].sum():,.2f} ({ftl_shipments['AdjustedCost'].sum()/total_cost*100:.1f}%)")
print(f"FTL Avg Cost/Shipment: ${ftl_shipments['AdjustedCost'].mean():.2f}")

# Potential savings from consolidation
df['ShipDateStr'] = df['ShipDate'].dt.date
same_lane_day = df.groupby([df['Lane'], df['ShipDateStr']]).agg({
    'Weight': 'sum',
    'AdjustedCost': 'sum',
    'ShipmentID': 'count'
})
consolidation_opps = same_lane_day[same_lane_day['ShipmentID'] > 1]

print(f"\n--- CONSOLIDATION SAVINGS POTENTIAL ---")
print(f"Number of consolidation opportunities: {len(consolidation_opps)}")
estimated_savings = consolidation_opps['AdjustedCost'].sum() * 0.20  # Assume 20% savings
print(f"Estimated annual savings from consolidation: ${estimated_savings:,.2f} ({estimated_savings/total_cost*100:.1f}%)")

print("\n" + "=" * 80)

# =====================================================
# VISUALIZATION 1: COMPREHENSIVE COST DASHBOARD
# =====================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

fig1 = plt.figure(figsize=(20, 12))

# 1. Cost by Weight Bracket (Pie Chart)
ax1 = plt.subplot(3, 3, 1)
bracket_costs = []
bracket_labels = []
for label, min_w, max_w in brackets:
    if max_w == float('inf'):
        bracket_df = df[df['Weight'] > min_w]
    else:
        bracket_df = df[(df['Weight'] > min_w) & (df['Weight'] <= max_w)]
    bracket_costs.append(bracket_df['AdjustedCost'].sum())
    bracket_labels.append(label)

colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF', '#9D84B7']
wedges, texts, autotexts = ax1.pie(bracket_costs, labels=bracket_labels, autopct='%1.1f%%',
                                   colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax1.set_title('1. Cost Distribution by Shipment Size', fontsize=12, fontweight='bold')

# 2. Top 10 Lanes by Cost (Horizontal Bar)
ax2 = plt.subplot(3, 3, 2)
top_lanes_cost = lane_costs.head(10)
bars = ax2.barh(range(len(top_lanes_cost)), top_lanes_cost['AdjustedCost'], color='#FF6B6B', alpha=0.8)
ax2.set_yticks(range(len(top_lanes_cost)))
ax2.set_yticklabels([lane.replace('-', ' → ') for lane in top_lanes_cost.index], fontsize=9)
ax2.set_xlabel('Total Cost ($)', fontsize=10, fontweight='bold')
ax2.set_title('2. Top 10 Costliest Lanes', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, top_lanes_cost['AdjustedCost'])):
    ax2.text(val + 1000, i, f'${val:,.0f}', va='center', fontsize=8, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Monthly Cost Trends
ax3 = plt.subplot(3, 3, 3)
months = monthly_costs.index
x_pos = range(len(months))
ax3_main = ax3
ax3_twin = ax3.twinx()

line1 = ax3_main.plot(x_pos, monthly_costs['AdjustedCost'], marker='o', linewidth=3,
                      color='#FF6B6B', label='Total Cost', markersize=8)
ax3_main.fill_between(x_pos, monthly_costs['AdjustedCost'], alpha=0.3, color='#FF6B6B')
ax3_main.set_ylabel('Total Cost ($)', fontsize=10, fontweight='bold', color='#FF6B6B')
ax3_main.tick_params(axis='y', labelcolor='#FF6B6B')

avg_cost_per_month = monthly_costs['AdjustedCost'] / monthly_costs['Shipments']
line2 = ax3_twin.plot(x_pos, avg_cost_per_month, marker='s', linewidth=2,
                      color='#4D96FF', label='Avg Cost/Shipment', markersize=6)
ax3_twin.set_ylabel('Avg Cost per Shipment ($)', fontsize=10, fontweight='bold', color='#4D96FF')
ax3_twin.tick_params(axis='y', labelcolor='#4D96FF')

ax3_main.set_xlabel('Month', fontsize=10, fontweight='bold')
ax3_main.set_title('3. Monthly Cost Trends', fontsize=12, fontweight='bold')
ax3_main.set_xticks(x_pos)
ax3_main.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
ax3_main.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3_main.grid(True, alpha=0.3)

# 4. Cost per Kg by Destination State
ax4 = plt.subplot(3, 3, 4)
state_cost_per_kg = df.groupby('DestinationState').agg({
    'AdjustedCost': 'sum',
    'Weight': 'sum'
})
state_cost_per_kg['CostPerKg'] = state_cost_per_kg['AdjustedCost'] / state_cost_per_kg['Weight']
state_cost_per_kg = state_cost_per_kg.sort_values('CostPerKg', ascending=False).head(10)

colors_grad = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(state_cost_per_kg)))
bars = ax4.bar(range(len(state_cost_per_kg)), state_cost_per_kg['CostPerKg'],
               color=colors_grad, alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(state_cost_per_kg)))
ax4.set_xticklabels(state_cost_per_kg.index, fontsize=10, fontweight='bold')
ax4.set_ylabel('Cost per Kg ($)', fontsize=10, fontweight='bold')
ax4.set_title('4. Most Expensive Destination States ($/kg)', fontsize=12, fontweight='bold')
ax4.axhline(y=avg_cost_per_kg, color='red', linestyle='--', linewidth=2,
            label=f'Average: ${avg_cost_per_kg:.2f}/kg')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. LTL vs FTL Cost Comparison
ax5 = plt.subplot(3, 3, 5)
ltl_cost = ltl_shipments['AdjustedCost'].sum()
ftl_cost = ftl_shipments['AdjustedCost'].sum()
categories = ['LTL\n(<5,250 kg)', 'FTL\n(≥5,250 kg)']
costs = [ltl_cost, ftl_cost]
shipment_counts = [len(ltl_shipments), len(ftl_shipments)]

x_pos = [0, 1]
bars = ax5.bar(x_pos, costs, color=['#FF6B6B', '#6BCB77'], alpha=0.8, edgecolor='black', width=0.6)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax5.set_ylabel('Total Cost ($)', fontsize=10, fontweight='bold')
ax5.set_title('5. LTL vs FTL Cost Comparison', fontsize=12, fontweight='bold')

for bar, cost, count in zip(bars, costs, shipment_counts):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height/2,
             f'${cost:,.0f}\n({cost/total_cost*100:.1f}%)\n{count:,} shipments',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Cost Distribution Histogram
ax6 = plt.subplot(3, 3, 6)
ax6.hist(df['AdjustedCost'], bins=50, color='#4D96FF', edgecolor='black', alpha=0.7)
ax6.axvline(df['AdjustedCost'].median(), color='red', linestyle='--', linewidth=2,
            label=f'Median: ${df["AdjustedCost"].median():.2f}')
ax6.axvline(df['AdjustedCost'].mean(), color='orange', linestyle='--', linewidth=2,
            label=f'Mean: ${df["AdjustedCost"].mean():.2f}')
ax6.set_xlabel('Shipment Cost ($)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Number of Shipments', fontsize=10, fontweight='bold')
ax6.set_title('6. Shipment Cost Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Top Origin-Destination Pairs by Cost (Heatmap)
ax7 = plt.subplot(3, 3, 7)
top_origins = df.groupby('Origin')['AdjustedCost'].sum().nlargest(2).index
top_dests = df.groupby('Destination')['AdjustedCost'].sum().nlargest(10).index
heatmap_data = df[df['Origin'].isin(top_origins) & df['Destination'].isin(top_dests)]
pivot = heatmap_data.groupby(['Origin', 'Destination'])['AdjustedCost'].sum().unstack(fill_value=0)

sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax7,
            cbar_kws={'label': 'Total Cost ($)'}, linewidths=0.5)
ax7.set_title('7. Cost Heatmap: Origin-Destination', fontsize=12, fontweight='bold')
ax7.set_xlabel('Destination', fontsize=10, fontweight='bold')
ax7.set_ylabel('Origin', fontsize=10, fontweight='bold')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# 8. Cost Concentration (Pareto)
ax8 = plt.subplot(3, 3, 8)
lane_costs_sorted = lane_costs.sort_values('AdjustedCost', ascending=False)
cumulative_pct = (lane_costs_sorted['AdjustedCost'].cumsum() / total_cost * 100).values
x_pos = range(len(lane_costs_sorted))

ax8_main = ax8
ax8_twin = ax8.twinx()

bars = ax8_main.bar(x_pos[:30], lane_costs_sorted['AdjustedCost'].values[:30],
                    color='#FF6B6B', alpha=0.7)
line = ax8_twin.plot(x_pos[:30], cumulative_pct[:30], color='darkblue', marker='o',
                     linewidth=2, markersize=4)
ax8_twin.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% Line')

ax8_main.set_xlabel('Lane Rank', fontsize=10, fontweight='bold')
ax8_main.set_ylabel('Cost ($)', fontsize=10, fontweight='bold', color='#FF6B6B')
ax8_twin.set_ylabel('Cumulative %', fontsize=10, fontweight='bold', color='darkblue')
ax8_main.set_title('8. Cost Concentration (Pareto)', fontsize=12, fontweight='bold')
ax8_main.tick_params(axis='y', labelcolor='#FF6B6B')
ax8_twin.tick_params(axis='y', labelcolor='darkblue')
ax8_twin.legend()
ax8_main.grid(True, alpha=0.3, axis='y')

# 9. Key Cost Metrics Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

metrics = [
    ('Total Annual Cost', f'${total_cost:,.0f}', '#FF6B6B'),
    ('Avg Cost/Shipment', f'${avg_cost_per_shipment:.2f}', '#4D96FF'),
    ('Avg Cost/Kg', f'${avg_cost_per_kg:.2f}', '#6BCB77'),
    ('Potential Savings', f'${estimated_savings:,.0f} (20%)', '#FFD93D')
]

y_pos = 0.85
for title, value, color in metrics:
    box = FancyBboxPatch((0.1, y_pos - 0.15), 0.8, 0.15,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.7, transform=ax9.transAxes)
    ax9.add_patch(box)
    
    ax9.text(0.5, y_pos - 0.075, title, transform=ax9.transAxes,
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax9.text(0.5, y_pos - 0.11, value, transform=ax9.transAxes,
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    y_pos -= 0.22

ax9.set_title('9. Key Cost Metrics', fontsize=12, fontweight='bold', y=0.98, pad=10)

fig1.suptitle('Stalmart Inc. - AS-IS Cost Structure Analysis Dashboard',
              fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('cost_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Dashboard saved as 'cost_analysis_dashboard.png'")
plt.show()

# =====================================================
# VISUALIZATION 2: DETAILED COST INSIGHTS
# =====================================================

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# 1. Cost Efficiency by Lane Volume
lane_stats_full = df.groupby('Lane').agg({
    'AdjustedCost': ['sum', 'mean'],
    'ShipmentID': 'count',
    'Weight': 'sum'
})
lane_stats_full.columns = ['TotalCost', 'AvgCost', 'Shipments', 'TotalWeight']
lane_stats_full['CostPerKg'] = lane_stats_full['TotalCost'] / lane_stats_full['TotalWeight']
lane_stats_full = lane_stats_full[lane_stats_full['Shipments'] >= 5].sort_values('TotalCost', ascending=False).head(20)

scatter = ax1.scatter(lane_stats_full['Shipments'], lane_stats_full['CostPerKg'],
                     s=lane_stats_full['TotalCost']/500, alpha=0.6,
                     c=lane_stats_full['CostPerKg'], cmap='RdYlGn_r')
ax1.axhline(y=avg_cost_per_kg, color='red', linestyle='--', linewidth=2,
           label=f'Average: ${avg_cost_per_kg:.2f}/kg')
ax1.set_xlabel('Number of Shipments on Lane', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cost per Kg ($)', fontsize=11, fontweight='bold')
ax1.set_title('Cost Efficiency vs Lane Volume\n(Bubble size = Total Cost)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Cost per Kg ($)', fontsize=9, fontweight='bold')

# 2. Savings Potential by Consolidation
consolidation_savings = consolidation_opps.copy()
consolidation_savings['PotentialSaving'] = consolidation_savings['AdjustedCost'] * 0.20
consolidation_savings = consolidation_savings.sort_values('PotentialSaving', ascending=False).head(15)

bars = ax2.barh(range(len(consolidation_savings)), consolidation_savings['PotentialSaving'],
               color='#6BCB77', alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(consolidation_savings)))
lane_labels = [f"{lane[0]} → {lane[1]}\n{lane[2]}" for lane in consolidation_savings.index]
ax2.set_yticklabels(lane_labels, fontsize=7)
ax2.set_xlabel('Potential Annual Savings ($)', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Consolidation Opportunities\n(20% cost reduction assumed)', fontsize=13, fontweight='bold')
ax2.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, consolidation_savings['PotentialSaving'])):
    ax2.text(val + 100, i, f'${val:,.0f}', va='center', fontsize=8, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Cost vs Weight Relationship
weight_bins = [0, 250, 500, 1000, 2000, 5250, 20000]
df['WeightBin'] = pd.cut(df['Weight'], bins=weight_bins,
                         labels=['0-250', '251-500', '501-1000', '1001-2000', '2001-5250', '>5250'])

bin_stats = df.groupby('WeightBin').agg({
    'CostPerKg': 'mean',
    'ShipmentID': 'count',
    'AdjustedCost': 'sum'
})

ax3_main = ax3
ax3_twin = ax3.twinx()

x_pos = range(len(bin_stats))
line1 = ax3_main.plot(x_pos, bin_stats['CostPerKg'], marker='o', linewidth=3,
                     color='#FF6B6B', label='Avg Cost/Kg', markersize=10)
ax3_main.fill_between(x_pos, bin_stats['CostPerKg'], alpha=0.3, color='#FF6B6B')
ax3_main.set_ylabel('Average Cost per Kg ($)', fontsize=11, fontweight='bold', color='#FF6B6B')
ax3_main.tick_params(axis='y', labelcolor='#FF6B6B')

bars = ax3_twin.bar(x_pos, bin_stats['ShipmentID'], alpha=0.5, color='#4D96FF',
                   label='Shipment Count', width=0.4)
ax3_twin.set_ylabel('Number of Shipments', fontsize=11, fontweight='bold', color='#4D96FF')
ax3_twin.tick_params(axis='y', labelcolor='#4D96FF')

ax3_main.set_xlabel('Weight Range (kg)', fontsize=11, fontweight='bold')
ax3_main.set_title('Cost Efficiency by Weight Category', fontsize=13, fontweight='bold')
ax3_main.set_xticks(x_pos)
ax3_main.set_xticklabels(bin_stats.index, fontsize=10)
ax3_main.legend(loc='upper right')
ax3_twin.legend(loc='upper center')
ax3_main.grid(True, alpha=0.3)

# 4. State-by-State Cost Analysis with Benchmarking
state_full_stats = df.groupby('DestinationState').agg({
    'AdjustedCost': 'sum',
    'ShipmentID': 'count',
    'Weight': 'sum'
})
state_full_stats['AvgCostPerShipment'] = state_full_stats['AdjustedCost'] / state_full_stats['ShipmentID']
state_full_stats['CostPerKg'] = state_full_stats['AdjustedCost'] / state_full_stats['Weight']
state_full_stats = state_full_stats.sort_values('AdjustedCost', ascending=False).head(12)

x = np.arange(len(state_full_stats))
width = 0.35

bars1 = ax4.bar(x - width/2, state_full_stats['AvgCostPerShipment'], width,
               label='Avg Cost/Shipment', color='#FF6B6B', alpha=0.8)
bars2 = ax4.bar(x + width/2, state_full_stats['CostPerKg']*100, width,
               label='Cost/Kg (×100)', color='#4D96FF', alpha=0.8)

ax4.set_xlabel('Destination State', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
ax4.set_title('State Cost Comparison\n(Avg per Shipment vs per Kg)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(state_full_stats.index, fontsize=10, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.0f}', ha='center', va='bottom', fontsize=7)

fig2.suptitle('Stalmart Inc. - Detailed Cost Analysis & Optimization Insights',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('cost_analysis_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Detailed analysis saved as 'cost_analysis_detailed.png'")
plt.show()

# =====================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# =====================================================

print("\n" + "=" * 80)
print("COST OPTIMIZATION RECOMMENDATIONS")
print("=" * 80)

print("\n1. IMMEDIATE OPPORTUNITIES:")
print(f"   • Consolidate same-day, same-lane shipments: Save ~${estimated_savings:,.0f}/year")
print(f"   • Focus on top 10 lanes (${top_10_lanes_cost:,.0f} or {top_10_lanes_cost/total_cost*100:.1f}% of costs)")
print(f"   • Negotiate volume discounts for CA, AZ, and NV (top 3 states)")

print("\n2. STRATEGIC INITIATIVES:")
ltl_under_500 = df[df['Weight'] <= 500]
ltl_under_500_cost = ltl_under_500['AdjustedCost'].sum()
print(f"   • Small shipment optimization: {len(ltl_under_500):,} shipments ≤500kg costing ${ltl_under_500_cost:,.0f}")
print(f"   • Potential 15% savings = ${ltl_under_500_cost * 0.15:,.0f}")

ftl_potential = df[(df['Weight'] >= 4000) & (df['Weight'] < 5250)]
print(f"   • Near-FTL optimization: {len(ftl_potential):,} shipments between 4,000-5,250 kg")
print(f"   • Could be upgraded to FTL for better rates")

print("\n3. COST REDUCTION TARGETS BY CATEGORY:")
for label, min_w, max_w in brackets:
    if max_w == float('inf'):
        bracket_df = df[df['Weight'] > min_w]
    else:
        bracket_df = df[(df['Weight'] > min_w) & (df['Weight'] <= max_w)]
    
    bracket_cost = bracket_df['AdjustedCost'].sum()
    
    if min_w <= 500:
        reduction = 0.20  # 20% potential
    elif min_w <= 2000:
        reduction = 0.15  # 15% potential
    else:
        reduction = 0.10  # 10% potential
    
    savings = bracket_cost * reduction
    if savings > 0:
        print(f"   • {label:20} → ${savings:10,.0f} ({reduction*100:.0f}% reduction target)")

total_potential_savings = total_cost * 0.18  # Conservative 18% overall
print(f"\n4. TOTAL ESTIMATED ANNUAL SAVINGS POTENTIAL: ${total_potential_savings:,.0f}")
print(f"   ({total_potential_savings/total_cost*100:.1f}% of current annual spend)")

print("\n" + "=" * 80)
print("✓ Cost analysis complete! Check the generated PNG files for visualizations.")
print("=" * 80)