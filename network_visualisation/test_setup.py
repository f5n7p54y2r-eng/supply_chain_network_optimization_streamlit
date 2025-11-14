#!/usr/bin/env python3
"""
Simple test script to verify your setup before running the dashboard
"""

import sys

print("🔍 Testing Network Visualization Setup...\n")

# Test 1: Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor} (OK)")
else:
    print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor} (Need 3.8+)")
    sys.exit(1)

# Test 2: Required packages
print("\n2. Checking required packages...")
required_packages = ['streamlit', 'pandas', 'plotly', 'networkx', 'numpy', 'openpyxl']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} (missing)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\nTo install, run:")
    print("   pip3 install -r requirements.txt")
    sys.exit(1)

# Test 3: Excel file
print("\n3. Checking Excel file...")
import os
excel_file = 'Network_modelling_assignment_copy.xlsx'
if os.path.exists(excel_file):
    print(f"   ✅ {excel_file} found")
    
    # Test 4: Load data
    print("\n4. Testing data load...")
    try:
        import pandas as pd
        df = pd.read_excel(excel_file, sheet_name='ShipmentData')
        
        # Set first row as column names (matching dashboard logic)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        
        print(f"   ✅ Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ['Origin State', 'DestinationState', 'Weight', 'ShipDate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing columns: {', '.join(missing_cols)}")
            print(f"   Available columns: {', '.join(df.columns.tolist())}")
        else:
            print(f"   ✅ All required columns present")
            
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
        sys.exit(1)
else:
    print(f"   ❌ {excel_file} not found")
    print("\n   Make sure the Excel file is in the same folder as this script.")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED! You're ready to run the dashboard.")
print("="*60)
print("\nTo start the dashboard, run:")
print("   streamlit run network_dashboard.py")
print("\nThe dashboard will open at: http://localhost:8501")
