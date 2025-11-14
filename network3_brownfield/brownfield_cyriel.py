"""
Crossdock Location Optimization using Mixed Integer Linear Programming (MILP)
This code determines which crossdocks to open to minimize total logistics costs
while considering consolidation benefits (FTL vs LTL rates)
"""

"""
recommended: create venv environment in root directory:
rm -rf ../venv(if exists)
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r ../requirements.txt
then run: python3 brownfield_cyriel.py
"""

import pandas as pd
import numpy as np
from pulp import *
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING ====================
print("="*80)
print("CROSSDOCK LOCATION OPTIMIZATION")
print("="*80)
print("\nLoading data...")

file_path = 'Brownfield_Study.xlsx'  # Update this path

# Load shipment data
shipments = pd.read_excel(file_path, sheet_name='ShipmentData')
print(f"Total shipments: {len(shipments):,}")
print(f"Total weight: {shipments['Weight'].sum():,.0f} kg")

# Define constants
suppliers = ['GA30043', 'CA91720']
crossdocks = ['NC27695', 'NY10006', 'TX75477', 'GA30113', 'IL61849']
FTL_RATE_PER_KM = 1.0
FTL_CAPACITY = 7000
FTL_MIN_LOAD = 5250

# Load distance matrices
dist_raw = pd.read_excel(file_path, sheet_name='Distance matrices', header=None)

# Parse distances
dist_supplier_customer = {}
dist_supplier_crossdock = {}
dist_crossdock_customer = {}

for row_idx in range(7, len(dist_raw)):
    # Supplier to customer
    customer = dist_raw.iloc[row_idx, 4]
    if pd.notna(customer):
        customer_str = str(customer).replace('US', '')
        for j, supplier in enumerate(suppliers):
            dist_val = dist_raw.iloc[row_idx, 5 + j]
            if pd.notna(dist_val):
                dist_supplier_customer[(supplier, customer_str)] = float(dist_val)
    
    # Supplier to crossdock
    crossdock = dist_raw.iloc[row_idx, 9]
    if pd.notna(crossdock):
        crossdock_str = str(crossdock).replace('US', '')
        for j, supplier in enumerate(suppliers):
            dist_val = dist_raw.iloc[row_idx, 10 + j]
            if pd.notna(dist_val):
                dist_supplier_crossdock[(supplier, crossdock_str)] = float(dist_val)
    
    # Crossdock to customer
    customer = dist_raw.iloc[row_idx, 14]
    if pd.notna(customer):
        customer_str = str(customer).replace('US', '')
        for j, crossdock in enumerate(crossdocks):
            dist_val = dist_raw.iloc[row_idx, 15 + j]
            if pd.notna(dist_val):
                dist_crossdock_customer[(crossdock, customer_str)] = float(dist_val)

print(f"Distance matrices loaded:")
print(f"  Supplier-Customer pairs: {len(dist_supplier_customer)}")
print(f"  Supplier-Crossdock pairs: {len(dist_supplier_crossdock)}")
print(f"  Crossdock-Customer pairs: {len(dist_crossdock_customer)}")

# Load crossdock costs
xdock_cost_raw = pd.read_excel(file_path, sheet_name='Crossdock Fixed and Variable Co', header=None)
fixed_cost = {}
variable_cost = {}

for i in range(3, 8):
    location = xdock_cost_raw.iloc[i, 1]
    if pd.notna(location) and str(location) in crossdocks:
        fixed_cost[str(location)] = float(xdock_cost_raw.iloc[i, 2])

for i in range(14, 19):
    location = xdock_cost_raw.iloc[i, 1]
    if pd.notna(location) and str(location) in crossdocks:
        variable_cost[str(location)] = float(xdock_cost_raw.iloc[i, 2])

print(f"\nCrossdock costs:")
for xdock in crossdocks:
    print(f"  {xdock}: Fixed=${fixed_cost.get(xdock, 0):,.0f}, Variable=${variable_cost.get(xdock, 0):.2f}/kg")

# Load LTL rates
ltl_raw = pd.read_excel(file_path, sheet_name='LTL Card Rate', header=None)

def get_ltl_rate(weight, distance):
    """Get LTL rate per kg per km based on weight and distance"""
    # Weight brackets
    if weight <= 500:
        w_col = 2
    elif weight <= 1000:
        w_col = 3
    elif weight <= 2000:
        w_col = 4
    elif weight <= 5000:
        w_col = 5
    else:
        w_col = 6
    
    # Distance brackets
    if distance <= 50:
        d_row = 2
    elif distance <= 250:
        d_row = 3
    elif distance <= 500:
        d_row = 4
    elif distance <= 1000:
        d_row = 5
    elif distance <= 1500:
        d_row = 6
    else:
        d_row = 7
    
    return float(ltl_raw.iloc[d_row, w_col])

# ==================== CONSOLIDATION ANALYSIS ====================
print("\n" + "="*80)
print("IDENTIFYING CONSOLIDATION OPPORTUNITIES")
print("="*80)

# Group shipments by date and origin for consolidation analysis
daily_origin_groups = shipments.groupby(['ShipDate', 'Origin']).agg({
    'Weight': 'sum',
    'ShipmentID': 'count'
}).reset_index()

print(f"\nDaily origin groups: {len(daily_origin_groups)}")

# Build routing options for each daily group
routing_options = []

for idx, group in daily_origin_groups.iterrows():
    date = group['ShipDate']
    origin = group['Origin']
    total_weight = group['Weight']
    
    # Get all shipments for this date-origin combination
    day_shipments = shipments[(shipments['ShipDate'] == date) & (shipments['Origin'] == origin)]
    
    # Calculate direct cost (baseline)
    direct_cost = 0
    for _, ship in day_shipments.iterrows():
        dest = ship['Destination']
        weight = ship['Weight']
        if (origin, dest) in dist_supplier_customer:
            dist = dist_supplier_customer[(origin, dest)]
            if weight >= FTL_MIN_LOAD:
                # Use FTL
                direct_cost += FTL_RATE_PER_KM * dist * np.ceil(weight / FTL_CAPACITY)
            else:
                # Use LTL
                ltl_rate = get_ltl_rate(weight, dist)
                direct_cost += ltl_rate * weight * dist
    
    # Add direct routing option
    routing_options.append({
        'group_id': idx,
        'date': date,
        'origin': origin,
        'option': 'DIRECT',
        'crossdock': None,
        'cost': direct_cost
    })
    
    # Calculate cost via each crossdock
    for xdock in crossdocks:
        if (origin, xdock) not in dist_supplier_crossdock:
            continue
        
        dist_to_xdock = dist_supplier_crossdock[(origin, xdock)]
        
        # Inbound leg: supplier to crossdock (consolidated)
        if total_weight >= FTL_MIN_LOAD:
            cost_inbound = FTL_RATE_PER_KM * dist_to_xdock * np.ceil(total_weight / FTL_CAPACITY)
        else:
            ltl_rate = get_ltl_rate(total_weight, dist_to_xdock)
            cost_inbound = ltl_rate * total_weight * dist_to_xdock
        
        # Outbound leg: crossdock to each destination
        cost_outbound = 0
        for _, ship in day_shipments.iterrows():
            dest = ship['Destination']
            weight = ship['Weight']
            if (xdock, dest) in dist_crossdock_customer:
                dist_from = dist_crossdock_customer[(xdock, dest)]
                if weight >= FTL_MIN_LOAD:
                    cost_outbound += FTL_RATE_PER_KM * dist_from * np.ceil(weight / FTL_CAPACITY)
                else:
                    ltl_rate = get_ltl_rate(weight, dist_from)
                    cost_outbound += ltl_rate * weight * dist_from
        
        # Processing cost
        processing_cost = variable_cost.get(xdock, 0) * total_weight
        
        # Total cost via this crossdock
        total_xdock_cost = cost_inbound + cost_outbound + processing_cost
        
        routing_options.append({
            'group_id': idx,
            'date': date,
            'origin': origin,
            'option': f'VIA_{xdock}',
            'crossdock': xdock,
            'cost': total_xdock_cost
        })

options_df = pd.DataFrame(routing_options)
print(f"Total routing options: {len(options_df)}")

# ==================== OPTIMIZATION MODEL ====================
print("\n" + "="*80)
print("BUILDING OPTIMIZATION MODEL")
print("="*80)

model = LpProblem("Crossdock_Optimization", LpMinimize)

# Decision variables
# y[k] = 1 if crossdock k is opened, 0 otherwise
y = {k: LpVariable(f"open_{k}", cat='Binary') for k in crossdocks}

# x[i] = 1 if routing option i is selected, 0 otherwise
x = {i: LpVariable(f"route_{i}", cat='Binary') for i in options_df.index}

print(f"Decision variables: {len(y)} crossdock + {len(x)} routing decisions")

# Constraint 1: Each shipment group must use exactly one routing option
for group_id in options_df['group_id'].unique():
    group_options = options_df[options_df['group_id'] == group_id].index
    model += (lpSum([x[i] for i in group_options]) == 1, f"one_route_{group_id}")

# Constraint 2: Can only route via a crossdock if it's open
for i, row in options_df.iterrows():
    if row['crossdock'] is not None:
        model += (x[i] <= y[row['crossdock']], f"open_req_{i}")

# Objective: Minimize total cost
total_cost = 0

# Fixed costs for opening crossdocks
for k in crossdocks:
    total_cost += fixed_cost.get(k, 0) * y[k]

# Routing costs (transportation + processing)
for i, row in options_df.iterrows():
    total_cost += row['cost'] * x[i]

model += total_cost

# Solve
print("\nSolving optimization model...")
print("(This may take a few minutes)")

solver = PULP_CBC_CMD(msg=True, timeLimit=600)
status = model.solve(solver)

# ==================== RESULTS ====================
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

print(f"\nSolution Status: {LpStatus[status]}")

if status != 1:
    print("ERROR: Could not find optimal solution")
    exit()

total_cost_value = value(model.objective)
print(f"\n{'='*80}")
print(f"TOTAL OPTIMIZED COST: ${total_cost_value:,.2f}")
print(f"{'='*80}")

# Which crossdocks to open
print("\n--- CROSSDOCK OPENING DECISIONS ---\n")
opened_xdocks = []
for k in crossdocks:
    if value(y[k]) > 0.5:
        opened_xdocks.append(k)
        print(f"✅ OPEN {k}")
        print(f"   • Fixed Cost: ${fixed_cost.get(k, 0):,.2f}")
        print(f"   • Variable Cost: ${variable_cost.get(k, 0):.2f}/kg")
    else:
        print(f"❌ DO NOT open {k}")

total_fixed = sum(fixed_cost.get(k, 0) * value(y[k]) for k in crossdocks)

# Routing statistics
print("\n--- ROUTING DECISIONS ---\n")

direct_routes = 0
xdock_routes = {k: 0 for k in crossdocks}
direct_weight = 0
xdock_weight = {k: 0 for k in crossdocks}

for i, row in options_df.iterrows():
    if value(x[i]) > 0.5:
        if row['option'] == 'DIRECT':
            direct_routes += 1
            group_data = daily_origin_groups[daily_origin_groups.index == row['group_id']].iloc[0]
            direct_weight += group_data['Weight']
        else:
            xdock = row['crossdock']
            xdock_routes[xdock] += 1
            group_data = daily_origin_groups[daily_origin_groups.index == row['group_id']].iloc[0]
            xdock_weight[xdock] += group_data['Weight']

print(f"Direct routes: {direct_routes} groups ({direct_weight:,.0f} kg)")
for k in opened_xdocks:
    print(f"Via {k}: {xdock_routes[k]} groups ({xdock_weight[k]:,.0f} kg)")

# Cost breakdown
variable_cost_total = total_cost_value - total_fixed
print(f"\n--- COST BREAKDOWN ---\n")
print(f"Fixed Costs: ${total_fixed:,.2f}")
print(f"Variable Costs (transport + processing): ${variable_cost_total:,.2f}")

# Calculate baseline (all direct)
baseline_cost = options_df[options_df['option'] == 'DIRECT'].groupby('group_id')['cost'].first().sum()
savings = baseline_cost - total_cost_value
savings_pct = (savings / baseline_cost) * 100

print(f"\n--- COMPARISON TO BASELINE ---\n")
print(f"Baseline (all direct, no crossdocks): ${baseline_cost:,.2f}")
print(f"Optimized cost: ${total_cost_value:,.2f}")
print(f"Total Savings: ${savings:,.2f}")
print(f"Savings Percentage: {savings_pct:.2f}%")

# ROI analysis
print(f"\n--- RETURN ON INVESTMENT ---\n")
for k in opened_xdocks:
    fc = fixed_cost.get(k, 0)
    # Calculate savings attributed to this crossdock
    direct_baseline = 0
    xdock_cost = 0
    for i, row in options_df.iterrows():
        if value(x[i]) > 0.5 and row['crossdock'] == k:
            # Find the direct cost for this group
            direct_row = options_df[(options_df['group_id'] == row['group_id']) & 
                                   (options_df['option'] == 'DIRECT')].iloc[0]
            direct_baseline += direct_row['cost']
            xdock_cost += row['cost']
    
    savings_for_xdock = direct_baseline - xdock_cost
    net_benefit = savings_for_xdock - fc
    roi = (net_benefit / fc) * 100 if fc > 0 else 0
    
    print(f"{k}:")
    print(f"  Fixed Cost: ${fc:,.2f}")
    print(f"  Variable Savings: ${savings_for_xdock:,.2f}")
    print(f"  Net Benefit: ${net_benefit:,.2f}")
    print(f"  ROI: {roi:.2f}%")
    print()

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create decision summary
decisions = []
for i, row in options_df.iterrows():
    if value(x[i]) > 0.5:
        decisions.append({
            'Date': row['date'],
            'Origin': row['origin'],
            'Routing': row['option'],
            'Crossdock': row['crossdock'] if row['crossdock'] else 'N/A',
            'Cost': row['cost']
        })

decisions_df = pd.DataFrame(decisions)
decisions_df.to_csv('routing_decisions.csv', index=False)
print("✓ Saved: routing_decisions.csv")

# Create crossdock summary
xdock_summary = []
for k in crossdocks:
    xdock_summary.append({
        'Crossdock': k,
        'Decision': 'OPEN' if k in opened_xdocks else 'DO NOT OPEN',
        'Fixed_Cost': fixed_cost.get(k, 0),
        'Variable_Cost_per_kg': variable_cost.get(k, 0),
        'Routes_Using': xdock_routes.get(k, 0),
        'Total_Weight_kg': xdock_weight.get(k, 0)
    })

xdock_summary_df = pd.DataFrame(xdock_summary)
xdock_summary_df.to_csv('crossdock_summary.csv', index=False)
print("✓ Saved: crossdock_summary.csv")

# Create executive summary
exec_summary = pd.DataFrame([{
    'Baseline_Cost': baseline_cost,
    'Optimized_Cost': total_cost_value,
    'Total_Savings': savings,
    'Savings_Percent': savings_pct,
    'Crossdocks_to_Open': len(opened_xdocks),
    'Opened_Crossdocks': ', '.join(opened_xdocks),
    'Total_Fixed_Investment': total_fixed
}])

exec_summary.to_csv('executive_summary.csv', index=False)
print("✓ Saved: executive_summary.csv")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nRecommendation: {'OPEN ' + ', '.join(opened_xdocks) if opened_xdocks else 'DO NOT OPEN any crossdocks'}")
if opened_xdocks:
    print(f"This will save ${savings:,.2f} ({savings_pct:.2f}%) annually")