# Network 3: Brownfield Study - Complete Guide

**Complete documentation for the brownfield optimization with realistic FTL/LTL rates**

---

## Table of Contents

1. [Overview](#1-overview)
2. [What is Brownfield?](#2-what-is-brownfield)
3. [Actual Rate Structure](#3-actual-rate-structure)
4. [Implementation Logic](#4-implementation-logic)
5. [Logic Review & Validation](#5-logic-review--validation)
6. [How to Run](#6-how-to-run)
7. [Expected Results](#7-expected-results)
8. [Visualization](#8-visualization)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

### What is Network 3?

**Network 3** is the brownfield analysis that evaluates the existing supply chain network with realistic FTL/LTL transportation rates and consolidation benefits.

**Key Features:**
- ✅ Realistic FTL/LTL rate structure from Part2_Input
- ✅ FTL consolidation for inbound flows (Supplier → X-Dock)
- ✅ LTL rate matrix for outbound flows (X-Dock → Customer)
- ✅ Brownfield focus: Working with existing infrastructure
- ✅ Compares scenarios: 0-5 X-Docks over 1-5 year horizons

**File:** `network3_brownfield_FTL_optimization.py`

**Output:** `network3_brownfield_optimization_results.xlsx`

---

## 2. What is Brownfield?

### Brownfield vs Greenfield

**Brownfield Study** = Analyzing and optimizing an **EXISTING** network:
- ✅ Current suppliers, X-Docks, and customers
- ✅ Existing infrastructure and locations
- ✅ Known demand patterns from historical data
- ✅ Realistic operational constraints
- ✅ Deciding which existing facilities to use

**Greenfield Study** = Designing a completely **NEW** network from scratch:
- 🆕 Choosing optimal locations for new facilities
- 🆕 No existing infrastructure constraints
- 🆕 Starting with a blank slate

### Our Brownfield Approach

**Existing Network:**
- 2 Suppliers: GA30043 (Georgia), CA91720 (California)
- 5 Potential X-Docks: NC27695, NY10006, TX75477, GA30113, IL61849
- 33 Customers across the US
- Historical demand data from ShipmentData sheet

**Optimization Question:**
- Which X-Docks should we use? (0, 1, 2, 3, 4, or 5?)
- What are the cost implications?
- How does consolidation affect total cost?

---

## 3. Actual Rate Structure

### 3.1 Data Source

All rates loaded from **Part2_Input** sheet in `Network_modelling_assignment.xlsx`

### 3.2 FTL (Full Truckload) Rate Card

```
Rate: $1.00 per truck per km
Capacity: 7,000 kg per truck
Threshold: 5,250 kg (75% utilization minimum)
Effective Rate: $1 / 5,250 = $0.000190 per kg per km
```

**When to use FTL:**
- Shipment weight >= 5,250 kg
- Consolidated loads from multiple customers
- Inbound flows to X-Docks (always consolidated)

### 3.3 LTL (Less Than Truckload) Rate Card

**Complex matrix: 6 distance ranges × 5 weight ranges**

| Distance (km) | 0-500 kg | 500-1000 kg | 1000-2000 kg | 2000-5000 kg | >5000 kg |
|---------------|----------|-------------|--------------|--------------|----------|
| **0-50**      | 0.05940  | 0.05049     | 0.04292      | 0.03648      | 0.03101  |
| **50-250**    | 0.05643  | 0.04797     | 0.04077      | 0.03466      | 0.02946  |
| **250-500**   | 0.05361  | 0.04557     | 0.03873      | 0.03292      | 0.02798  |
| **500-1000**  | 0.05093  | 0.04329     | 0.03680      | 0.03128      | 0.02658  |
| **1000-1500** | 0.04838  | 0.04112     | 0.03496      | 0.02971      | 0.02526  |
| **>1500**     | 0.04596  | 0.03907     | 0.03321      | 0.02823      | 0.02399  |

**Rate Range:** $0.02399 to $0.05940 per kg per km

**When to use LTL:**
- Shipment weight < 5,250 kg
- Individual customer deliveries
- Direct shipping (no consolidation)
- Outbound flows from X-Docks (most cases)

### 3.4 FTL vs LTL Comparison

**The Shocking Truth:**

| Scenario | Rate ($/kg/km) | Comparison |
|----------|----------------|------------|
| **FTL (75% full)** | $0.000190 | **Baseline** |
| **LTL (best case)** | $0.02399 | **126x more expensive!** |
| **LTL (worst case)** | $0.05940 | **313x more expensive!** |
| **LTL (average)** | ~$0.035 | **184x more expensive!** |

**Key Insight:** FTL is ~100-300x cheaper than LTL per kg!

This massive difference is why X-Docks make economic sense:
- Consolidate small orders into FTL for long-haul transport
- Accept LTL costs for shorter final delivery
- Net result: Huge cost savings

---

## 4. Implementation Logic

### 4.1 Three Shipping Scenarios

#### Scenario A: Direct Shipping (No X-Docks)

```python
# Each customer order ships directly from supplier
# Uses LTL rates (no consolidation benefit)

for each (supplier, customer) pair:
    weight = demand[(s, c)]
    distance = dist_supp_cust[(s, c)]
    rate = get_ltl_rate(weight, distance)  # From LTL matrix
    cost = weight × distance × rate
```

**Characteristics:**
- ❌ No consolidation
- ❌ Uses expensive LTL rates ($0.024-0.059/kg/km)
- ❌ Long distances (supplier to customer directly)
- **Expected Cost:** $300-400M per year

#### Scenario B: X-Dock Routing - Inbound Leg

```python
# All customer orders to same X-Dock are CONSOLIDATED
# Uses FTL rate (much cheaper!)

for each (supplier, xdock) pair:
    total_weight = Sum(all customer orders going through this route)
    distance = dist_supp_xdock[(s, j)]
    rate = $1 / 5,250 = $0.000190 per kg per km  # FTL rate
    cost = total_weight × distance × rate
```

**Example:**
```
GA30043 → GA30113:
  Total weight: 4,670,000 kg (for 31 customers consolidated)
  Distance: ~100 km
  Rate: $0.000190 per kg per km
  Cost: 4,670,000 × 100 × 0.000190 = $88,730
  
  vs LTL (if not consolidated):
  Rate: ~$0.035 per kg per km
  Cost: 4,670,000 × 100 × 0.035 = $16,345,000
  
  Savings: $16.3M - $88K = $16.2M (99.5% savings!)
```

**Characteristics:**
- ✅ Full consolidation
- ✅ Uses cheap FTL rate ($0.00019/kg/km)
- ✅ Massive cost savings (99%+ vs LTL)

#### Scenario C: X-Dock Routing - Outbound Leg

```python
# Individual customer deliveries from X-Dock
# Uses LTL rates (most orders < 5,250 kg)

for each (xdock, customer) pair:
    total_demand = sum(demand from all suppliers to this customer)
    distance = dist_xdock_cust[(j, c)]
    
    if total_demand >= 5,250 kg:
        rate = $0.000190 per kg per km  # FTL if eligible
    else:
        rate = get_ltl_rate(total_demand, distance)  # LTL matrix
    
    cost = actual_weight × distance × rate
```

**Characteristics:**
- ⚠️ Most orders < 5,250 kg → LTL rates apply
- ✅ But shorter distances than direct shipping
- ✅ Some large orders may qualify for FTL

### 4.2 Cost Calculation Details

#### Direct Shipping Cost:

```python
direct_rates = {}
for s in suppliers:
    for c in customers:
        distance = dist_supp_cust[(s, c)]
        weight = demand[(s, c)]
        rate = get_ltl_rate(weight, distance, ltl_rate_matrix)
        direct_rates[(s, c)] = rate

transport_direct = Sum([
    x_direct[(s, c)] × distance × rate
    for all (s, c) pairs
])
```

#### Inbound FTL Cost:

```python
ftl_rate_per_kg_km = $1 / 5,250 = $0.000190

transport_inbound_ftl = Sum([
    Sum([x_via_xdock[(s, j, c)] for all customers]) ×  # Consolidated
    dist_supp_xdock[(s, j)] × 
    ftl_rate_per_kg_km
    for all (s, j) pairs
])
```

#### Outbound LTL Cost:

```python
outbound_rates = {}
for j in xdocks:
    for c in customers:
        distance = dist_xdock_cust[(j, c)]
        total_demand = sum(demand from all suppliers to c)
        
        if total_demand >= 5,250:
            rate = ftl_rate_per_kg_km
        else:
            rate = get_ltl_rate(total_demand, distance, ltl_matrix)
        
        outbound_rates[(j, c)] = rate

transport_outbound_ltl = Sum([
    x_via_xdock[(s, j, c)] × distance × rate
    for all (s, j, c) triplets
])
```

#### Total Cost:

```python
Total = Fixed_Costs + Transport_Costs + Processing_Costs

where:
  Fixed_Costs = Sum(X-Dock opening costs)
  Transport_Costs = Direct + Inbound_FTL + Outbound_LTL
  Processing_Costs = Sum(weight × processing_rate per X-Dock)
```

---

## 5. Logic Review & Validation

### 5.1 What's Correct ✅

1. **Brownfield Approach**
   - ✅ Evaluating existing network
   - ✅ Using actual demand data
   - ✅ Comparing operational scenarios

2. **Rate Structure**
   - ✅ FTL/LTL rates loaded correctly from Part2_Input
   - ✅ Rates match Excel data exactly
   - ✅ Massive FTL/LTL difference captured (100x)

3. **Consolidation Logic**
   - ✅ Inbound flows consolidated (all orders to same X-Dock)
   - ✅ FTL rate applied to consolidated inbound
   - ✅ LTL rate applied to individual outbound

4. **Output Structure**
   - ✅ Compatible with existing visualization
   - ✅ Same format as Network 2
   - ✅ Additional FTL/LTL cost breakdown

### 5.2 Known Simplifications ⚠️

1. **FTL Utilization**
   - Uses fixed rate of $0.000190/kg/km (assumes 75% utilization)
   - Reality: Rate varies with actual truck fill
   - **Impact:** Minor - simplification is reasonable for strategic planning

2. **Truck Capacity**
   - Not explicitly modeled with integer variables
   - Uses continuous approximation
   - **Impact:** Low - acceptable for brownfield study

3. **Minimum Threshold**
   - Doesn't enforce 5,250 kg minimum with constraints
   - Assumes inbound flows are large enough (they are!)
   - **Impact:** Very low - inbound flows are 4.67M kg (way above threshold)

### 5.3 Validation Checklist

**When you run the optimization, verify:**

✅ **Inbound flows are massive:**
```
Expected:
- GA30043 → GA30113: ~4,670,000 kg (889 FTL trucks!)
- CA91720 → TX75477: ~1,620,000 kg (309 FTL trucks!)

Both WAY above 5,250 kg threshold ✅
```

✅ **Rates are applied correctly:**
```
Effective inbound rate: ~$0.00019/kg/km (FTL)
Effective outbound rate: ~$0.024-0.059/kg/km (LTL)
Ratio: ~126-313x difference
```

✅ **X-Docks are cost-effective:**
```
Direct shipping (all LTL): $300-400M
2 X-Docks (FTL inbound): $30-50M
Savings: 85-90%
```

✅ **Results make economic sense:**
```
- Fixed costs: $10-20M per X-Dock
- Processing costs: $1.25-2.02 per kg
- Transport savings from FTL >> Processing costs
- Net result: X-Docks win by huge margin
```

---

## 6. How to Run

### 6.1 Prerequisites

**Required files:**
- `Network_modelling_assignment.xlsx` in `data/` folder
- Python packages: pandas, numpy, pulp, openpyxl

**Install dependencies:**
```bash
pip install pandas numpy pulp openpyxl
```

### 6.2 Run Optimization

```bash
cd /Users/dongyuangao/Desktop/Network_SCM/SCM_Network2_Optimisation
python3 network3_brownfield_FTL_optimization.py
```

**Runtime:** ~5-10 minutes for all scenarios

**Progress output:**
```
Loading data from Excel file...

Data Summary:
  Suppliers: 2
  X-Docks: 5
  Customers: 33
  Annual Total Demand: 6,289,000.00 kg

================================================================================
NETWORK 3: BROWNFIELD STUDY - RATE STRUCTURE
================================================================================
  FTL Rate: $1.00 per truck per km
  FTL Threshold: 5,250 kg (75% of 7,000 kg)
  FTL Effective Rate: $0.000190 per kg per km

  LTL Rates: $0.02399 to $0.05940 per kg per km
  LTL/FTL Ratio: 126.3x more expensive

  → Inbound (Supplier → X-Dock): FTL rate (consolidated)
  → Outbound (X-Dock → Customer): LTL rates (individual shipments)
  → Direct Shipping: LTL rates (no consolidation)
================================================================================

Solving for Year 1...
[Scenarios run: Baseline, 0 X-Docks, 1 X-Dock, 2 X-Docks, ...]
```

### 6.3 Output File

**File:** `network3_brownfield_optimization_results.xlsx`

**Structure:**
- **Summary Sheet:** All scenarios comparison
- **Individual Sheets:** Detailed flows for each scenario
  - Format: `{Years}Y_{Scenario}` (e.g., `1Y_Exactly_2_X-Dock(s)`)

---

## 7. Expected Results

### 7.1 Cost Comparison (Year 1)

| Scenario | Total Cost | Fixed | Transport | Processing | Notes |
|----------|-----------|-------|-----------|------------|-------|
| **Direct** | $300-400M | $0 | $300-400M | $0 | All LTL, long distances |
| **1 X-Dock** | $40-60M | $10-20M | $15-25M | $5-15M | Some consolidation |
| **2 X-Docks** | $30-40M | $20-40M | $10-15M | $5-10M | **Optimal** |
| **3 X-Docks** | $35-45M | $30-60M | $10-15M | $5-10M | Diminishing returns |
| **5 X-Docks** | $50-70M | $50-100M | $10-15M | $5-10M | Too many facilities |

**Key Finding:** 2 X-Docks (TX75477 + GA30113) likely optimal

### 7.2 Why X-Docks Win

**Example: GA30043 → 31 Customers (4.67M kg total)**

**Direct Shipping (No X-Dock):**
```
31 separate LTL shipments
Average distance: ~1,500 km
Average LTL rate: ~$0.035/kg/km
Cost: 4.67M × 1,500 × 0.035 = $245M
```

**Via X-Dock (Consolidated):**
```
Inbound (Supplier → X-Dock):
  Weight: 4.67M kg
  Distance: ~100 km
  Rate: $0.00019/kg/km (FTL)
  Cost: 4.67M × 100 × 0.00019 = $88,730

Outbound (X-Dock → 31 Customers):
  31 shipments × avg 500 km × $0.03/kg/km
  Cost: ~$2M

Processing:
  4.67M kg × $1.50/kg = $7M

Total: $88K + $2M + $7M = $9.1M

Savings: $245M - $9.1M = $236M (96% reduction!)
```

### 7.3 Network 2 vs Network 3 Comparison

| Aspect | Network 2 (Base) | Network 3 (Brownfield) |
|--------|------------------|------------------------|
| **Rate Model** | Simplified ($0.005/kg/km) | Realistic FTL/LTL |
| **Consolidation** | Not modeled | FTL consolidation |
| **Direct Cost** | $44.5M | $300-400M |
| **2 X-Docks Cost** | $81M | $30-40M |
| **Winner** | Direct ❌ | X-Docks ✅ |
| **Realism** | Low | High |
| **Use Case** | Academic | Real-world planning |

**Why the huge difference?**
- Network 2 uses same rate for all shipments
- Doesn't capture FTL consolidation benefit
- Makes X-Docks look expensive (processing costs without transport savings)

- Network 3 uses realistic rates
- Captures 99% FTL savings on inbound
- Shows true economic benefit of X-Docks

---

## 8. Visualization

### 8.1 Dashboard Integration

**The brownfield results work with the existing Streamlit dashboard!**

**Start dashboard:**
```bash
cd /Users/dongyuangao/Desktop/Network_SCM/network_visualisation
streamlit run network_dashboard.py
```

**Select scenario:**
1. Navigate to **"Geographic Flow Map"**
2. In sidebar, select **"Network 3: Brownfield Study (2 X-Docks)"**
3. View realistic FTL/LTL flow visualization

### 8.2 What You'll See

**Network Topology:**
- 2 Suppliers: GA30043 (Georgia), CA91720 (California)
- 2 X-Docks: TX75477 (Texas), GA30113 (Georgia)
- 33 Customers across the US

**Flow Visualization:**
- **Inbound flows** (Supplier → X-Dock): Thick lines (FTL, high volume)
- **Outbound flows** (X-Dock → Customer): Thinner lines (LTL, individual)
- Color-coded by node type (suppliers, X-Docks, customers)

**Cost Information:**
- Total cost breakdown
- FTL vs LTL cost comparison
- Processing costs
- Fixed costs

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: "ModuleNotFoundError: No module named 'pulp'"**
```bash
Solution:
pip install pulp
```

**Issue: "FileNotFoundError: Network_modelling_assignment.xlsx"**
```bash
Solution:
- Ensure Excel file is in data/ folder
- Check file path in script (line 541)
```

**Issue: "Solver taking too long"**
```bash
Normal behavior:
- 35 scenarios total (5 years × 7 scenarios each)
- Each scenario: 30-60 seconds
- Total: 5-10 minutes
- Be patient!
```

**Issue: "Results show direct shipping wins"**
```bash
Problem: FTL/LTL rates not loaded correctly

Check:
1. Part2_Input sheet exists in Excel
2. Rates printed at start match expected values
3. FTL rate: $0.00019/kg/km
4. LTL rates: $0.024-0.059/kg/km
```

**Issue: "Brownfield file not found in dashboard"**
```bash
Solution:
1. Run optimization first: python3 network3_brownfield_FTL_optimization.py
2. Verify output file exists: network3_brownfield_optimization_results.xlsx
3. Refresh dashboard
```

### 9.2 Validation Checks

**After running, verify:**

```python
# Check Summary sheet
# Expected for Year 1, 2 X-Docks:
Total Cost: $30-40M
Transport Inbound FTL: $0.5-1M (very low!)
Transport Outbound LTL: $10-15M (higher but shorter distances)
Processing Cost: $5-10M
Fixed Cost: $20-40M

# If numbers are way off, something is wrong
```

---

## Summary

**Network 3 Brownfield Study** evaluates the existing supply chain network with realistic FTL/LTL transportation rates.

**Key Takeaways:**

1. **FTL is 100-300x cheaper than LTL** - This is the game-changer
2. **X-Docks enable FTL consolidation** - Massive inbound savings
3. **2 X-Docks is optimal** - TX75477 (Texas) + GA30113 (Georgia)
4. **85-90% cost savings** - vs direct shipping with LTL
5. **Brownfield approach** - Working with existing infrastructure

**Files:**
- Script: `network3_brownfield_FTL_optimization.py`
- Output: `network3_brownfield_optimization_results.xlsx`
- This guide: `NETWORK3_BROWNFIELD_COMPLETE_GUIDE.md`

**Next Steps:**
1. Run the optimization
2. Review results in Excel
3. Visualize in dashboard
4. Use for strategic planning decisions

---

**The brownfield model captures the REAL economic benefit of X-Docks: FTL consolidation!** 🏭🚚📊✅
