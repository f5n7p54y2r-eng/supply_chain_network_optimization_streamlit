"""
Network 3: Brownfield Study - MILP Optimization with FTL Consolidation.

This is the brownfield analysis that evaluates the existing network with realistic
transportation rates, including FTL consolidation benefits at X-Docks.

Brownfield Study Characteristics:
- Evaluates EXISTING network with actual FTL/LTL rate structure
- Models consolidation: Supplier → X-Dock uses FTL rates ($1/truck/km)
- Individual deliveries: X-Dock → Customer uses LTL rate matrix
- Compares scenarios: 0-5 X-Docks over 1-5 year horizons

Key Features:
1. FTL consolidation for inbound flows (Supplier → X-Dock)
2. LTL rate matrix for outbound flows (X-Dock → Customer)
3. Realistic cost structure from Part2_Input rate cards
4. Brownfield focus: Working with existing infrastructure constraints
"""

import pandas as pd
import numpy as np
from pulp import *
import time
from pathlib import Path

# ============================================================================
# TRANSPORTATION RATE STRUCTURE (Loaded from Part2_Input)
# ============================================================================
# FTL (Full Truckload):
#   - Rate: $1.00 per truck per km (from Part2_Input)
#   - Capacity: 7,000 kg per truck
#   - Threshold: 5,250 kg (75% utilization)
#   - Effective rate: $1/5,250 = $0.00019 per kg per km
#
# LTL (Less Than Truckload):
#   - Complex rate matrix based on weight AND distance
#   - Range: $0.024 to $0.059 per kg per km
#   - Loaded from Part2_Input LTL Rate Card
#
# Key Insight:
#   - FTL is ~100x cheaper than LTL per kg!
#   - Inbound consolidation (Supplier → X-Dock) uses FTL
#   - Outbound (X-Dock → Customer) uses LTL (most orders < 5,250 kg)
# ============================================================================


def _extract_zip(code):
    """Utility to pull digit-only ZIP strings from supplier/X-dock/customer IDs."""
    if code is None:
        return None
    digits = ''.join(ch for ch in str(code) if ch.isdigit())
    return digits or None


def load_data(file_path):
    """Load all data from Excel file"""

    # Read Part1_Input sheet with proper structure
    df_part1 = pd.read_excel(file_path, sheet_name='Part1_Input', header=None)

    # Extract Suppliers
    suppliers = ['GA30043', 'CA91720']

    # Extract X-Docks
    xdocks = ['NC27695', 'NY10006', 'TX75477', 'GA30113', 'IL61849']

    # Extract Customers (demand points) - rows 7-34, column 4
    customers = []
    for i in range(7, 35):
        cust = df_part1.iloc[i, 4]
        if pd.notna(cust):
            customers.append(str(cust))

    # Extract Distance: Supplier to Customer (columns 5-6, rows 7-34)
    dist_supp_cust = {}
    for i, cust in enumerate(customers):
        row_idx = 7 + i
        dist_supp_cust[(suppliers[0], cust)] = float(df_part1.iloc[row_idx, 5])
        dist_supp_cust[(suppliers[1], cust)] = float(df_part1.iloc[row_idx, 6])

    # Extract Distance: Supplier to X-Dock (columns 10-11, rows 7-11)
    dist_supp_xdock = {}
    for i, xdock in enumerate(xdocks):
        row_idx = 7 + i
        dist_supp_xdock[(suppliers[0], xdock)] = float(df_part1.iloc[row_idx, 10])
        dist_supp_xdock[(suppliers[1], xdock)] = float(df_part1.iloc[row_idx, 11])

    # Extract Distance: X-Dock to Customer (columns 15-19, rows 7-34)
    dist_xdock_cust = {}
    for i, cust in enumerate(customers):
        row_idx = 7 + i
        for j, xdock in enumerate(xdocks):
            col_idx = 15 + j  # Start from column 15, not 14
            dist_xdock_cust[(xdock, cust)] = float(df_part1.iloc[row_idx, col_idx])

    # Extract Fixed Cost of Opening X-Dock (column 23, rows 3-7)
    fixed_cost_xdock = {}
    for i, xdock in enumerate(xdocks):
        row_idx = 3 + i
        fixed_cost_xdock[xdock] = float(df_part1.iloc[row_idx, 23])

    # Extract Variable Cost - Inventory Processing Cost (column 23, rows 14-18)
    var_cost_xdock = {}
    for i, xdock in enumerate(xdocks):
        row_idx = 14 + i
        var_cost_xdock[xdock] = float(df_part1.iloc[row_idx, 23])

    # Extract Variable Cost - Transportation (per kg per km) (row 24, column 21)
    transport_cost_per_kg_km = float(df_part1.iloc[24, 21])
    
    # ========================================================================
    # Load FTL and LTL Rate Cards from Part2_Input
    # ========================================================================
    df_part2 = pd.read_excel(file_path, sheet_name='Part2_Input', header=None)
    
    # FTL Rate Card
    ftl_rate_per_truck_km = float(df_part2.iloc[13, 2])  # $1 per truck per km
    ftl_capacity_kg = 7000  # From row 17
    ftl_threshold_kg = 5250  # 75% of capacity, from row 19
    
    # LTL Rate Card (6 distance ranges × 5 weight ranges)
    ltl_rates = {}
    distance_ranges = [(0, 50), (50, 250), (250, 500), (500, 1000), (1000, 1500), (1500, float('inf'))]
    weight_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]
    
    for i, dist_range in enumerate(distance_ranges):
        for j, weight_range in enumerate(weight_ranges):
            row_idx = 5 + i  # LTL rates start at row 5
            col_idx = 4 + j  # Weight columns start at column 4
            rate = float(df_part2.iloc[row_idx, col_idx])
            ltl_rates[(dist_range, weight_range)] = rate
    # ========================================================================

    # Extract Supplier Capacities (rows 28-29, column 23)
    supplier_capacity = {}
    supplier_capacity[suppliers[0]] = float(df_part1.iloc[28, 23])
    supplier_capacity[suppliers[1]] = float(df_part1.iloc[29, 23])

    # Read Shipment Data
    df_shipments = pd.read_excel(file_path, sheet_name='ShipmentData', header=1)

    # Create mapping from zip code to customer name (with state prefix)
    zip_to_customer = {}
    for cust in customers:
        if len(cust) >= 6:
            zip_str = cust[2:]  # Remove state prefix
        else:
            zip_str = cust

        try:
            zip_code = int(zip_str)
            zip_to_customer[zip_code] = cust
        except ValueError:
            pass

    # Aggregate annual demand by origin-destination pair
    demand_annual = df_shipments.groupby(['Origin Zip', 'DestinationZip'])['Weight'].sum().to_dict()

    demand = {}
    missing_customers = set()

    for (origin, dest), weight in demand_annual.items():
        origin_str = None
        origin_zip = str(int(origin))
        for sup in suppliers:
            sup_zip = sup[2:] if len(sup) >= 6 else sup
            if origin_zip == sup_zip:
                origin_str = sup
                break

        if origin_str is None:
            continue

        dest_int = int(dest)
        if dest_int in zip_to_customer:
            dest_str = zip_to_customer[dest_int]
            demand[(origin_str, dest_str)] = float(weight)
        else:
            dest_str = str(dest_int)
            if dest_str not in missing_customers:
                missing_customers.add(dest_str)
                customers.append(dest_str)

                avg_dist_supp_cust = np.mean([v for v in dist_supp_cust.values()])
                for sup in suppliers:
                    dist_supp_cust[(sup, dest_str)] = avg_dist_supp_cust

                avg_dist_xdock_cust = np.mean([v for v in dist_xdock_cust.values()])
                for xdock in xdocks:
                    dist_xdock_cust[(xdock, dest_str)] = avg_dist_xdock_cust

            demand[(origin_str, dest_str)] = float(weight)

    return {
        'suppliers': suppliers,
        'xdocks': xdocks,
        'customers': customers,
        'dist_supp_cust': dist_supp_cust,
        'dist_supp_xdock': dist_supp_xdock,
        'dist_xdock_cust': dist_xdock_cust,
        'fixed_cost_xdock': fixed_cost_xdock,
        'var_cost_xdock': var_cost_xdock,
        'transport_cost_per_kg_km': transport_cost_per_kg_km,
        'supplier_capacity': supplier_capacity,
        'demand': demand,
        'annual_total_demand': df_shipments['Weight'].sum(),
        # FTL/LTL rate data
        'ftl_rate_per_truck_km': ftl_rate_per_truck_km,
        'ftl_capacity_kg': ftl_capacity_kg,
        'ftl_threshold_kg': ftl_threshold_kg,
        'ltl_rates': ltl_rates,
        'distance_ranges': distance_ranges,
        'weight_ranges': weight_ranges
    }


def get_ltl_rate(weight_kg, distance_km, ltl_rates, distance_ranges, weight_ranges):
    """Get LTL rate based on weight and distance from rate card."""
    # Find distance range
    dist_range = None
    for dr in distance_ranges:
        if dr[0] <= distance_km < dr[1]:
            dist_range = dr
            break
    if dist_range is None:
        dist_range = distance_ranges[-1]  # Use last range if beyond
    
    # Find weight range
    weight_range = None
    for wr in weight_ranges:
        if wr[0] <= weight_kg < wr[1]:
            weight_range = wr
            break
    if weight_range is None:
        weight_range = weight_ranges[-1]  # Use last range if beyond
    
    return ltl_rates.get((dist_range, weight_range), ltl_rates[(distance_ranges[-1], weight_ranges[-1])])


def get_ftl_rate_per_kg_km(weight_kg, ftl_rate_per_truck_km, ftl_threshold_kg):
    """Calculate FTL rate per kg per km based on actual weight."""
    # FTL cost is fixed per truck, so rate per kg depends on utilization
    # Minimum weight for FTL is ftl_threshold_kg (5,250 kg)
    if weight_kg >= ftl_threshold_kg:
        # Use actual weight for rate calculation (better utilization = lower rate per kg)
        return ftl_rate_per_truck_km / weight_kg
    else:
        # Not eligible for FTL
        return None


def solve_milp(data, num_years, xdock_open_target=None, xdock_open_upper=None):
    """Solve the MILP for the given number of years and X-Dock constraints."""

    suppliers = data['suppliers']
    xdocks = data['xdocks']
    customers = data['customers']
    demand = data['demand']

    demand_scaled = {key: val * num_years for key, val in demand.items()}

    model = LpProblem(f"Network_Optimization_{num_years}Y", LpMinimize)

    y = LpVariable.dicts("XDock_Open", xdocks, cat='Binary')
    x_direct = LpVariable.dicts(
        "Direct_Flow",
        [(s, c) for s in suppliers for c in customers],
        lowBound=0,
        cat='Continuous'
    )
    x_via_xdock = LpVariable.dicts(
        "XDock_Flow",
        [(s, j, c) for s in suppliers for j in xdocks for c in customers],
        lowBound=0,
        cat='Continuous'
    )

    fixed_costs = lpSum([data['fixed_cost_xdock'][j] * y[j] for j in xdocks])

    # ========================================================================
    # REALISTIC FTL/LTL RATE STRUCTURE
    # ========================================================================
    
    # Pre-calculate rates for direct shipping (use LTL rates based on demand)
    direct_rates = {}
    for s in suppliers:
        for c in customers:
            if (s, c) in data['dist_supp_cust'] and (s, c) in demand_scaled:
                distance = data['dist_supp_cust'][(s, c)]
                weight = demand_scaled[(s, c)]
                # Direct shipping uses LTL rates (no consolidation)
                rate = get_ltl_rate(weight, distance, data['ltl_rates'], 
                                   data['distance_ranges'], data['weight_ranges'])
                direct_rates[(s, c)] = rate
    
    # Direct shipping cost
    transport_direct = lpSum([
        x_direct[(s, c)] * data['dist_supp_cust'][(s, c)] * direct_rates.get((s, c), 0.05)
        for s in suppliers for c in customers
        if (s, c) in data['dist_supp_cust']
    ])
    
    # ========================================================================
    # INBOUND: Supplier → X-Dock (CONSOLIDATED - USE FTL RATE)
    # ========================================================================
    # Key insight: All shipments from supplier S to X-Dock J are consolidated
    # Total weight will be HUGE (e.g., 4.67M kg), far exceeding FTL threshold
    # Use FTL rate: $1 per truck per km, with threshold of 5,250 kg
    
    # Calculate FTL rate per kg per km (assuming 75% utilization as minimum)
    ftl_rate_per_kg_km = data['ftl_rate_per_truck_km'] / data['ftl_threshold_kg']
    
    transport_inbound_ftl = lpSum([
        lpSum([x_via_xdock[(s, j, c)] for c in customers]) *  # Total consolidated flow S→J
        data['dist_supp_xdock'][(s, j)] * 
        ftl_rate_per_kg_km  # FTL rate: $1/5250 = $0.00019 per kg per km
        for s in suppliers for j in xdocks
        if (s, j) in data['dist_supp_xdock']
    ])
    
    # ========================================================================
    # OUTBOUND: X-Dock → Customer (USE LTL RATES)
    # ========================================================================
    # Individual customer shipments - use LTL rate matrix
    # Most shipments will be < 5,250 kg, so LTL applies
    
    # Pre-calculate LTL rates for each X-Dock → Customer flow
    outbound_rates = {}
    for j in xdocks:
        for c in customers:
            if (j, c) in data['dist_xdock_cust']:
                distance = data['dist_xdock_cust'][(j, c)]
                
                # Use ACTUAL total demand to customer (not average per supplier)
                # This represents the actual shipment size from X-Dock to customer
                total_to_customer = sum(demand_scaled.get((s, c), 0) for s in suppliers)
                
                # If no demand, use small default weight
                if total_to_customer == 0:
                    total_to_customer = 1000  # Default 1 ton
                
                # Check if eligible for FTL (total demand >= 5,250 kg)
                if total_to_customer >= data['ftl_threshold_kg']:
                    # Use FTL rate - large customer order can be consolidated
                    rate = ftl_rate_per_kg_km
                else:
                    # Use LTL rate from matrix based on actual demand
                    rate = get_ltl_rate(total_to_customer, distance, data['ltl_rates'],
                                       data['distance_ranges'], data['weight_ranges'])
                outbound_rates[(j, c)] = rate
    
    transport_outbound_ltl = lpSum([
        x_via_xdock[(s, j, c)] * 
        data['dist_xdock_cust'][(j, c)] * 
        outbound_rates.get((j, c), 0.03)  # Default to mid-range LTL rate
        for s in suppliers for j in xdocks for c in customers
        if (j, c) in data['dist_xdock_cust']
    ])
    
    # Total X-Dock transportation = Inbound (FTL) + Outbound (LTL)
    transport_via_xdock = transport_inbound_ftl + transport_outbound_ltl
    # ========================================================================

    processing_cost = lpSum([
        x_via_xdock[(s, j, c)] * data['var_cost_xdock'][j]
        for s in suppliers for j in xdocks for c in customers
    ])

    model += fixed_costs + transport_direct + transport_via_xdock + processing_cost, "Total_Cost"

    for s in suppliers:
        for c in customers:
            if (s, c) in demand_scaled:
                model += (
                    x_direct[(s, c)] +
                    lpSum([x_via_xdock[(s, j, c)] for j in xdocks])
                    == demand_scaled[(s, c)],
                    f"Demand_{s}_{c}"
                )
            else:
                model += x_direct[(s, c)] == 0, f"NoDemand_Direct_{s}_{c}"
                for j in xdocks:
                    model += x_via_xdock[(s, j, c)] == 0, f"NoDemand_XDock_{s}_{j}_{c}"

    for s in suppliers:
        model += (
            lpSum([x_direct[(s, c)] for c in customers]) +
            lpSum([x_via_xdock[(s, j, c)] for j in xdocks for c in customers])
            <= data['supplier_capacity'][s] * num_years,
            f"Supplier_Capacity_{s}"
        )

    M = data['annual_total_demand'] * num_years * 2
    for j in xdocks:
        model += (
            lpSum([x_via_xdock[(s, j, c)] for s in suppliers for c in customers])
            <= M * y[j],
            f"XDock_Opening_{j}"
        )

    # ----- Scenario control constraints --------------------------------------
    if xdock_open_target is not None:
        # Scenario control: enforce an exact number of X-docks
        model += (
            lpSum([y[j] for j in xdocks]) == xdock_open_target,
            f"XDock_Count_Target_{xdock_open_target}"
        )
        
        # FORCE all flows through X-Docks when X-Docks are required to be open
        # This ensures we're comparing X-Dock scenarios properly
        if xdock_open_target > 0:
            for s in suppliers:
                for c in customers:
                    model += x_direct[(s, c)] == 0, f"Force_XDock_Route_{s}_{c}"
    elif xdock_open_upper is not None:
        # Scenario control: cap the number of X-docks that may open
        model += (
            lpSum([y[j] for j in xdocks]) <= xdock_open_upper,
            f"XDock_Count_Upper_{xdock_open_upper}"
        )
    # ------------------------------------------------------------------------

    print(f"\n{'='*80}")
    print(f"Solving MILP for {num_years} year(s)...")
    print(f"{'='*80}")

    start_time = time.time()
    solver = PULP_CBC_CMD(msg=1, timeLimit=300)
    model.solve(solver)
    solve_time = time.time() - start_time

    # ----- Scenario labeling -------------------------------------------------
    scenario_label = 'Unconstrained'
    if xdock_open_target is not None:
        # Track exact-count scenario label in results
        scenario_label = f'Exactly {xdock_open_target} X-Dock(s)'
    elif xdock_open_upper is not None:
        # Track capped scenario label in results
        scenario_label = f'Up to {xdock_open_upper} X-Dock(s)'
    # ------------------------------------------------------------------------

    results = {
        'num_years': num_years,
        'status': LpStatus.get(model.status, str(model.status)),
        'solve_time': solve_time,
        'total_cost': value(model.objective),
        'fixed_cost': value(fixed_costs),
        'transport_direct': value(transport_direct),
        'transport_via_xdock': value(transport_via_xdock),
        'transport_inbound_ftl': value(transport_inbound_ftl),  # NEW: FTL inbound
        'transport_outbound_ltl': value(transport_outbound_ltl),  # NEW: LTL outbound
        'processing_cost': value(processing_cost),
        'xdocks_opened': [j for j in xdocks if value(y[j]) > 0.5],
        'direct_flow': {
            (s, c): value(x_direct[(s, c)])
            for s in suppliers for c in customers
            if value(x_direct[(s, c)]) > 0.01
        },
        'xdock_flow': {
            (s, j, c): value(x_via_xdock[(s, j, c)])
            for s in suppliers for j in xdocks for c in customers
            if value(x_via_xdock[(s, j, c)]) > 0.01
        },
        'scenario_label': scenario_label,
        'xdock_target': xdock_open_target,
        'xdock_upper': xdock_open_upper
    }

    return results


def print_results(results):
    """Print optimization results in a formatted way."""

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS - {results['num_years']} YEAR(S)")
    print(f"{'='*80}")
    print(f"Scenario: {results['scenario_label']}")
    print(f"Status: {results['status']}")
    print(f"Solve Time: {results['solve_time']:.2f} seconds")
    print(f"\nCost Breakdown:")
    print(f"  Fixed Cost (X-Dock opening):     ${results['fixed_cost']:,.2f}")
    print(f"  Transportation (Direct):         ${results['transport_direct']:,.2f}")
    print(f"  Transportation (Via X-Dock):     ${results['transport_via_xdock']:,.2f}")
    if 'transport_inbound_ftl' in results:
        print(f"    ├─ Inbound (FTL):              ${results['transport_inbound_ftl']:,.2f}")
        print(f"    └─ Outbound (LTL):             ${results['transport_outbound_ltl']:,.2f}")
    print(f"  Processing Cost (X-Dock):        ${results['processing_cost']:,.2f}")
    print(f"  {'─'*50}")
    print(f"  TOTAL COST:                      ${results['total_cost']:,.2f}")

    print(f"\nX-Docks Opened: {len(results['xdocks_opened'])}")
    for xdock in results['xdocks_opened']:
        print(f"  - {xdock}")

    print(f"\nDirect Shipments: {len(results['direct_flow'])}")
    total_direct = sum(results['direct_flow'].values())
    print(f"  Total Weight (Direct): {total_direct:,.2f} kg")

    print(f"\nX-Dock Shipments: {len(results['xdock_flow'])}")
    total_xdock = sum(results['xdock_flow'].values())
    print(f"  Total Weight (Via X-Dock): {total_xdock:,.2f} kg")

    total_flow = total_direct + total_xdock
    if total_flow > 0:
        print(f"\nPercentage via X-Dock: {100 * total_xdock / total_flow:.2f}%")
    else:
        print(f"\nPercentage via X-Dock: N/A (no flow)")


def create_summary_table(all_results):
    """Create a summary comparison table for all time horizons."""

    summary_data = []
    for results in all_results:
        row = {
            'Years': results['num_years'],
            'Scenario': results['scenario_label'],
            'Total Cost ($)': results['total_cost'],
            'Fixed Cost ($)': results['fixed_cost'],
            'Transport Direct ($)': results['transport_direct'],
            'Transport XDock ($)': results['transport_via_xdock'],
        }
        # Add FTL/LTL breakdown if available
        if 'transport_inbound_ftl' in results:
            row['Transport Inbound FTL ($)'] = results['transport_inbound_ftl']
            row['Transport Outbound LTL ($)'] = results['transport_outbound_ltl']
        row.update({
            'Processing Cost ($)': results['processing_cost'],
            'XDocks Opened': len(results['xdocks_opened']),
            'XDock Names': ', '.join(results['xdocks_opened']),
            'Solve Time (s)': results['solve_time']
        })
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))

    return df_summary


BASE_DIR = Path(__file__).resolve().parent


def main():
    """Main execution function"""

    file_path = BASE_DIR.parent / 'data' / 'Network_modelling_assignment.xlsx'

    print("Loading data from Excel file...")
    data = load_data(file_path)

    print(f"\nData Summary:")
    print(f"  Suppliers: {len(data['suppliers'])}")
    print(f"  X-Docks: {len(data['xdocks'])}")
    print(f"  Customers: {len(data['customers'])}")
    print(f"  Annual Total Demand: {data['annual_total_demand']:,.2f} kg")
    
    print(f"\n{'='*80}")
    print("NETWORK 3: BROWNFIELD STUDY - RATE STRUCTURE")
    print(f"{'='*80}")
    ftl_rate_per_kg = data['ftl_rate_per_truck_km'] / data['ftl_threshold_kg']
    print(f"  FTL Rate: ${data['ftl_rate_per_truck_km']:.2f} per truck per km")
    print(f"  FTL Threshold: {data['ftl_threshold_kg']:,.0f} kg (75% of {data['ftl_capacity_kg']:,.0f} kg)")
    print(f"  FTL Effective Rate: ${ftl_rate_per_kg:.6f} per kg per km")
    print(f"\n  LTL Rates: ${min(data['ltl_rates'].values()):.5f} to ${max(data['ltl_rates'].values()):.5f} per kg per km")
    print(f"  LTL/FTL Ratio: {min(data['ltl_rates'].values()) / ftl_rate_per_kg:.1f}x more expensive")
    print(f"\n  → Inbound (Supplier → X-Dock): FTL rate (consolidated)")
    print(f"  → Outbound (X-Dock → Customer): LTL rates (individual shipments)")
    print(f"  → Direct Shipping: LTL rates (no consolidation)")
    print(f"{'='*80}")

    all_results = []
    xdock_counts = list(range(0, len(data['xdocks']) + 1))

    for num_years in [1, 2, 3, 4, 5]:
        # ----- Scenario sweep ------------------------------------------------
        # Baseline: allow MILP to choose any number of X-docks
        baseline_results = solve_milp(data, num_years)
        print_results(baseline_results)
        all_results.append(baseline_results)

        # Scenario sweep: force every possible exact X-dock count
        for required_count in xdock_counts:
            scenario_results = solve_milp(data, num_years, xdock_open_target=required_count)
            print_results(scenario_results)
            all_results.append(scenario_results)
        # --------------------------------------------------------------------

    df_summary = create_summary_table(all_results)

    output_file = BASE_DIR / 'network3_brownfield_optimization_results.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        for results in all_results:
            sheet_name = f"{results['num_years']}Y_{results['scenario_label'].replace(' ', '_')[:25]}"
            sheet_name = sheet_name[:31]

            df_xdocks = pd.DataFrame([
                {
                    'XDock': xdock,
                    'XDock_Zip': _extract_zip(xdock),
                    'Status': 'Opened'
                }
                for xdock in results['xdocks_opened']
            ])

            df_direct = pd.DataFrame([
                {
                    'Supplier': s,
                    'Supplier_Zip': _extract_zip(s),
                    'Customer': c,
                    'Customer_Zip': _extract_zip(c),
                    'Flow_kg': flow
                }
                for (s, c), flow in results['direct_flow'].items()
            ])

            df_xdock_flow = pd.DataFrame([
                {
                    'Supplier': s,
                    'Supplier_Zip': _extract_zip(s),
                    'XDock': j,
                    'XDock_Zip': _extract_zip(j),
                    'Customer': c,
                    'Customer_Zip': _extract_zip(c),
                    'Flow_kg': flow
                }
                for (s, j, c), flow in results['xdock_flow'].items()
            ])

            df_summary_single = pd.DataFrame([{
                'Years': results['num_years'],
                'Scenario': results['scenario_label'],
                'Total Cost': results['total_cost'],
                'Fixed Cost': results['fixed_cost'],
                'Transport Direct': results['transport_direct'],
                'Transport XDock': results['transport_via_xdock'],
                'Processing Cost': results['processing_cost'],
                'Solve Time (s)': results['solve_time']
            }])

            df_summary_single.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            df_xdocks.to_excel(writer, sheet_name=sheet_name, startrow=3, index=False)

            if not df_direct.empty:
                df_direct.to_excel(writer, sheet_name=sheet_name,
                                   startrow=3 + len(df_xdocks) + 3, index=False)

            if not df_xdock_flow.empty:
                start_row = 3 + len(df_xdocks) + 3 + (len(df_direct) if not df_direct.empty else 0) + 3
                df_xdock_flow.to_excel(writer, sheet_name=sheet_name,
                                      startrow=start_row, index=False)

    print(f"\n{'='*80}")
    print(f"NETWORK 3: BROWNFIELD STUDY RESULTS")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"\nBrownfield Analysis Complete:")
    print(f"  - Evaluated existing network with realistic FTL/LTL rates")
    print(f"  - Scenarios: 0-5 X-Docks across 1-5 year horizons")
    print(f"  - FTL consolidation modeled for inbound flows")
    print(f"  - LTL rate matrix applied for outbound flows")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
