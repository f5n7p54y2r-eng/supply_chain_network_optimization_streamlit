import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

file_path = './data/Network_modelling_assignment.xlsx' 
df_raw = pd.read_excel(file_path, sheet_name='ShipmentData')

df = df_raw.iloc[1:].copy()
df.columns = ['ShipmentID', 'ShipDate', 'OriginZip', 'OriginState', 'Origin',
              'Destination', 'DestinationZip', 'DestinationState', 'Weight']

df['ShipmentID'] = pd.to_numeric(df['ShipmentID'], errors='coerce')
df['ShipDate'] = pd.to_datetime(df['ShipDate'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['OriginZip'] = pd.to_numeric(df['OriginZip'], errors='coerce')
df['DestinationZip'] = pd.to_numeric(df['DestinationZip'], errors='coerce')

df = df.dropna(subset=['Weight', 'Origin', 'Destination'])

print("=" * 80)
print("DESCRIPTIVE STATISTICS FOR SHIPMENT DATA")
print("=" * 80)


print("\n--- 1. OVERALL VOLUME METRICS ---")
total_shipments = len(df)
total_weight = df['Weight'].sum()
avg_weight = df['Weight'].mean()
min_weight = df['Weight'].min()
max_weight = df['Weight'].max()

print(f"Total number of shipments: {total_shipments:,}")
print(f"Total Weight: {total_weight:,.0f} kg")
print(f"Average Weight per Shipment: {avg_weight:.2f} kg")
print(f"Min Weight: {min_weight:.0f} kg")
print(f"Max Weight: {max_weight:.0f} kg")

min_date = df['ShipDate'].min()
max_date = df['ShipDate'].max()
days_diff = (max_date - min_date).days

print(f"\nDate Range: {min_date.date()} to {max_date.date()}")
print(f"Duration: {days_diff} days (~{days_diff/365:.1f} years)")
print(f"Average shipments per day: {total_shipments/days_diff:.1f}")

print("\n--- 2. WEIGHT DISTRIBUTION ---")

median = df['Weight'].median()
p25 = df['Weight'].quantile(0.25)
p75 = df['Weight'].quantile(0.75)
p90 = df['Weight'].quantile(0.90)
p95 = df['Weight'].quantile(0.95)
std_dev = df['Weight'].std()
cv = (std_dev / avg_weight) * 100

print(f"Median: {median:.0f} kg")
print(f"25th Percentile: {p25:.0f} kg")
print(f"75th Percentile: {p75:.0f} kg")
print(f"90th Percentile: {p90:.0f} kg")
print(f"95th Percentile: {p95:.0f} kg")
print(f"Standard Deviation: {std_dev:.2f} kg")
print(f"Coefficient of Variation: {cv:.1f}%")

print("\nWeight Distribution by Bracket:")
weight_brackets = [
    (0, 250, '0-250 kg'),
    (250, 500, '251-500 kg'),
    (500, 1000, '501-1000 kg'),
    (1000, 2000, '1001-2000 kg'),
    (2000, 5000, '2001-5000 kg'),
    (5000, float('inf'), '>5000 kg')
]

for min_w, max_w, label in weight_brackets:
    if max_w == float('inf'):
        count = len(df[df['Weight'] > min_w])
    else:
        count = len(df[(df['Weight'] > min_w) & (df['Weight'] <= max_w)])
    pct = (count / total_shipments) * 100
    print(f"  {label}: {count} shipments ({pct:.1f}%)")


print("\n--- 3. GEOGRAPHIC ANALYSIS ---")

unique_origins = df['Origin'].nunique()
unique_destinations = df['Destination'].nunique()
unique_origin_states = sorted(df['OriginState'].dropna().unique())
unique_dest_states = sorted(df['DestinationState'].dropna().unique())

print(f"Unique Origin Locations: {unique_origins}")
print(f"  Origin States ({len(unique_origin_states)}): {', '.join(unique_origin_states)}")
print(f"Unique Destination Locations: {unique_destinations}")
print(f"  Destination States ({len(unique_dest_states)}): {', '.join(unique_dest_states)}")

print("\nTop Origins by Shipment Count:")
origin_stats = df.groupby('Origin').agg({
    'ShipmentID': 'count',
    'Weight': 'sum',
    'OriginState': 'first'
}).rename(columns={'ShipmentID': 'Count', 'Weight': 'TotalWeight'})
origin_stats = origin_stats.sort_values('Count', ascending=False)

for idx, (origin, row) in enumerate(origin_stats.head(5).iterrows(), 1):
    pct = (row['Count'] / total_shipments) * 100
    print(f"  {idx}. {origin} ({row['OriginState']}): {row['Count']} shipments ({pct:.1f}%), {row['TotalWeight']:.0f} kg")

print("\nTop Destinations by Shipment Count:")
dest_stats = df.groupby('Destination').agg({
    'ShipmentID': 'count',
    'Weight': 'sum',
    'DestinationState': 'first'
}).rename(columns={'ShipmentID': 'Count', 'Weight': 'TotalWeight'})
dest_stats = dest_stats.sort_values('Count', ascending=False)

for idx, (dest, row) in enumerate(dest_stats.head(10).iterrows(), 1):
    pct = (row['Count'] / total_shipments) * 100
    print(f"  {idx}. {dest} ({row['DestinationState']}): {row['Count']} shipments ({pct:.1f}%), {row['TotalWeight']:.0f} kg")

print("\nTop Destination States:")
state_stats = df.groupby('DestinationState').agg({
    'ShipmentID': 'count',
    'Weight': 'sum'
}).rename(columns={'ShipmentID': 'Count', 'Weight': 'TotalWeight'})
state_stats = state_stats.sort_values('Count', ascending=False)

for idx, (state, row) in enumerate(state_stats.head(10).iterrows(), 1):
    pct = (row['Count'] / total_shipments) * 100
    print(f"  {idx}. {state}: {row['Count']} shipments ({pct:.1f}%), {row['TotalWeight']:.0f} kg")


print("\n--- 4. LANE ANALYSIS ---")

df['Lane'] = df['Origin'] + '-' + df['Destination']

lane_stats = df.groupby('Lane').agg({
    'ShipmentID': 'count',
    'Weight': ['sum', 'mean']
})
lane_stats.columns = ['Count', 'TotalWeight', 'AvgWeight']
lane_stats = lane_stats.sort_values('Count', ascending=False)

unique_lanes = len(lane_stats)
print(f"Total Unique Lanes: {unique_lanes}")

print("\nTop 15 Lanes by Shipment Count:")
for idx, (lane, row) in enumerate(lane_stats.head(15).iterrows(), 1):
    pct = (row['Count'] / total_shipments) * 100
    print(f"  {idx}. {lane}: {row['Count']} shipments ({pct:.1f}%), {row['TotalWeight']:.0f} kg total, {row['AvgWeight']:.0f} kg avg")

top10_lanes = lane_stats.head(10)['Count'].sum()
top20_lanes = lane_stats.head(20)['Count'].sum()

print(f"\nLane Concentration:")
print(f"  Top 10 lanes: {top10_lanes} shipments ({top10_lanes/total_shipments*100:.1f}%)")
print(f"  Top 20 lanes: {top20_lanes} shipments ({top20_lanes/total_shipments*100:.1f}%)")


print("\n--- 5. CONSOLIDATION OPPORTUNITY ANALYSIS ---")

df['ShipDateStr'] = df['ShipDate'].dt.date

same_day_origin = df.groupby([df['Origin'], df['ShipDateStr']]).size()
same_day_origin_multi = same_day_origin[same_day_origin > 1]
total_consolidatable_origin = same_day_origin_multi.sum()

print(f"Days with multiple shipments from same origin: {len(same_day_origin_multi)}")
print(f"Total shipments that could be consolidated at origin: {total_consolidatable_origin} ({total_consolidatable_origin/total_shipments*100:.1f}%)")

same_day_dest = df.groupby([df['Destination'], df['ShipDateStr']]).size()
same_day_dest_multi = same_day_dest[same_day_dest > 1]
total_consolidatable_dest = same_day_dest_multi.sum()

print(f"\nDays with multiple shipments to same destination: {len(same_day_dest_multi)}")
print(f"Total shipments that could be consolidated at destination: {total_consolidatable_dest} ({total_consolidatable_dest/total_shipments*100:.1f}%)")

same_lane_day = df.groupby([df['Lane'], df['ShipDateStr']]).size()
same_lane_day_multi = same_lane_day[same_lane_day > 1]
total_same_lane_day = same_lane_day_multi.sum()

print(f"\nSame lane, same day opportunities: {len(same_lane_day_multi)}")
print(f"Shipments on same lane, same day: {total_same_lane_day} ({total_same_lane_day/total_shipments*100:.1f}%)")

print("\nTop 10 Same-Day Lane Consolidation Opportunities:")
consolidation_opps = df.groupby([df['Lane'], df['ShipDateStr']]).agg({
    'ShipmentID': 'count',
    'Weight': 'sum'
}).rename(columns={'ShipmentID': 'Count', 'Weight': 'TotalWeight'})
consolidation_opps = consolidation_opps[consolidation_opps['Count'] > 1]
consolidation_opps = consolidation_opps.sort_values('Count', ascending=False)

for idx, ((lane, date), row) in enumerate(consolidation_opps.head(10).iterrows(), 1):
    print(f"  {idx}. {lane} on {date}: {row['Count']} shipments, {row['TotalWeight']:.0f} kg total")

print("\n" + "=" * 80)

# ============================================================================
# INDIVIDUAL GRAPH WINDOWS - FIRST SET (Dashboard graphs)
# ============================================================================

print("\nGenerating individual graph windows...")

# Graph 1: Top Destinations
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111)
top_dests = dest_stats.head(10).sort_values('Count')
colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_dests)))
bars = ax1.barh(range(len(top_dests)), top_dests['Count'], color=colors1, edgecolor='black')
ax1.set_yticks(range(len(top_dests)))
ax1.set_yticklabels(top_dests.index, fontsize=10, fontweight='bold')
ax1.set_xlabel('Number of Shipments', fontsize=11, fontweight='bold')
ax1.set_title('1. Top 10 Destinations by Shipment Count', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, top_dests['Count'])):
    ax1.text(val + 5, bar.get_y() + bar.get_height() / 2, f'{int(val)}',
             va='center', fontsize=9, fontweight='bold')
plt.tight_layout()

# Graph 2: Top Lanes
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
top_lanes_plot = lane_stats.head(10).sort_values('Count')
colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_lanes_plot)))
bars = ax2.barh(range(len(top_lanes_plot)), top_lanes_plot['Count'], color=colors2, edgecolor='black')
ax2.set_yticks(range(len(top_lanes_plot)))
ax2.set_yticklabels([lane.replace('-', ' → ') for lane in top_lanes_plot.index], fontsize=9)
ax2.set_xlabel('Number of Shipments', fontsize=11, fontweight='bold')
ax2.set_title('2. Top 10 Lanes by Shipment Count', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, top_lanes_plot['Count']):
    ax2.text(val + 3, bar.get_y() + bar.get_height() / 2, f'{int(val)}',
             va='center', fontsize=9, fontweight='bold')
plt.tight_layout()

# Graph 3: Weight Distribution Histogram
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
n, bins, patches = ax3.hist(df['Weight'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
cm = plt.cm.RdYlGn_r
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(i / len(patches)))
ax3.axvline(df['Weight'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["Weight"].mean():.0f} kg')
ax3.axvline(df['Weight'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {df["Weight"].median():.0f} kg')
ax3.set_xlabel('Weight (kg)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('3. Shipment Weight Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Graph 4: Monthly Shipment Trends
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)
df['YearMonth'] = df['ShipDate'].dt.to_period('M')
monthly_stats = df.groupby('YearMonth').agg({
    'ShipmentID': 'count',
    'Weight': 'sum'
})
monthly_stats.index = monthly_stats.index.to_timestamp()
ax4.plot(monthly_stats.index, monthly_stats['ShipmentID'], marker='o', linewidth=2,
         markersize=6, color='steelblue', label='Shipment Count')
ax4.fill_between(monthly_stats.index, monthly_stats['ShipmentID'], alpha=0.3, color='steelblue')
ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Shipments', fontsize=11, fontweight='bold')
ax4.set_title('4. Monthly Shipment Volume Trend', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Graph 5: Top Origins
fig5 = plt.figure(figsize=(8, 6))
ax5 = fig5.add_subplot(111)
top_origins = origin_stats.head(5).sort_values('Count')
colors5 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars = ax5.barh(range(len(top_origins)), top_origins['Count'], color=colors5, edgecolor='black', alpha=0.8)
ax5.set_yticks(range(len(top_origins)))
ax5.set_yticklabels(top_origins.index, fontsize=11, fontweight='bold')
ax5.set_xlabel('Number of Shipments', fontsize=11, fontweight='bold')
ax5.set_title('5. Top 5 Origins by Shipment Count', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, top_origins['Count']):
    ax5.text(val + 10, bar.get_y() + bar.get_height() / 2, f'{int(val)}',
             va='center', fontsize=10, fontweight='bold')
plt.tight_layout()

# Graph 6: Weight Categories Pie Chart
fig6 = plt.figure(figsize=(8, 6))
ax6 = fig6.add_subplot(111)
weight_categories = {
    'Very Light\n(0-250 kg)': len(df[df['Weight'] <= 250]),
    'Light\n(251-500 kg)': len(df[(df['Weight'] > 250) & (df['Weight'] <= 500)]),
    'Medium\n(501-1000 kg)': len(df[(df['Weight'] > 500) & (df['Weight'] <= 1000)]),
    'Heavy\n(1001-2000 kg)': len(df[(df['Weight'] > 1000) & (df['Weight'] <= 2000)]),
    'Very Heavy\n(2001-5000 kg)': len(df[(df['Weight'] > 2000) & (df['Weight'] <= 5000)]),
    'FTL Eligible\n(>5000 kg)': len(df[df['Weight'] > 5000])
}
colors6 = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = ax6.pie(weight_categories.values(), labels=weight_categories.keys(),
                                     autopct='%1.1f%%', startangle=90, colors=colors6,
                                     textprops={'fontsize': 9, 'fontweight': 'bold'})
ax6.set_title('6. Shipment Distribution by Weight Category', fontsize=13, fontweight='bold')
plt.tight_layout()

# Graph 7: Origin-Destination Heatmap
fig7 = plt.figure(figsize=(10, 8))
ax7 = fig7.add_subplot(111)
od_matrix = df.groupby(['Origin', 'Destination']).size().reset_index(name='Count')
pivot = od_matrix.pivot(index='Origin', columns='Destination', values='Count').fillna(0)
sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', ax=ax7,
            cbar_kws={'label': 'Shipment Count'}, linewidths=0.5)
ax7.set_title('7. Origin-Destination Shipment Heatmap', fontsize=13, fontweight='bold')
ax7.set_xlabel('Destination', fontsize=11, fontweight='bold')
ax7.set_ylabel('Origin', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Graph 8: Day of Week Distribution
fig8 = plt.figure(figsize=(8, 6))
ax8 = fig8.add_subplot(111)
df['DayOfWeek'] = df['ShipDate'].dt.dayofweek
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_stats = df.groupby('DayOfWeek').size()
colors_dow = ['#FF6B6B' if i >= 5 else '#4ECDC4' for i in range(7)]
bars = ax8.bar(range(7), [dow_stats.get(i, 0) for i in range(7)],
               color=colors_dow, alpha=0.8, edgecolor='black')
ax8.set_xticks(range(7))
ax8.set_xticklabels(day_names, fontsize=11, fontweight='bold')
ax8.set_ylabel('Number of Shipments', fontsize=11, fontweight='bold')
ax8.set_title('8. Day of Week Distribution', fontsize=13, fontweight='bold')
ax8.axhline(y=dow_stats.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Average: {dow_stats.mean():.0f}')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Graph 9: Consolidation Opportunity Summary
fig9 = plt.figure(figsize=(8, 6))
ax9 = fig9.add_subplot(111)
ax9.axis('off')

ftl_eligible = len(df[df['Weight'] >= 5250])
summary_data = [
    ('Same Lane, Same Day', f'{total_same_lane_day / total_shipments * 100:.1f}%',
     f'{total_same_lane_day} shipments', '#90EE90'),
    ('Origin Consolidation', f'{total_consolidatable_origin / total_shipments * 100:.1f}%',
     f'{total_consolidatable_origin} shipments', '#87CEEB'),
    ('Currently Using LTL', f'{(total_shipments - ftl_eligible) / total_shipments * 100:.1f}%',
     f'{total_shipments - ftl_eligible} shipments', '#FFB6C1'),
    ('FTL Eligible Now', f'{ftl_eligible / total_shipments * 100:.1f}%',
     f'{ftl_eligible} shipments', '#DDA0DD')
]

y_pos = 0.85
for title, pct, count, color in summary_data:
    box = FancyBboxPatch((0.1, y_pos - 0.15), 0.8, 0.15,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.7, transform=ax9.transAxes)
    ax9.add_patch(box)

    ax9.text(0.5, y_pos - 0.075, title, transform=ax9.transAxes,
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax9.text(0.5, y_pos - 0.11, f'{pct} ({count})', transform=ax9.transAxes,
             ha='center', va='center', fontsize=10)

    y_pos -= 0.22

ax9.set_title('9. Consolidation Opportunity Summary', fontsize=13, fontweight='bold',
              y=0.98, pad=10)
plt.tight_layout()

# ============================================================================
# INDIVIDUAL GRAPH WINDOWS - SECOND SET (Additional insights)
# ============================================================================

# Graph 10: Lane Concentration (Pareto Analysis)
fig10 = plt.figure(figsize=(10, 6))
ax10_main = fig10.add_subplot(111)
ax10_twin = ax10_main.twinx()

lane_stats_plot = df.groupby('Lane').size().sort_values(ascending=False)
cumulative_pct = (lane_stats_plot.cumsum() / lane_stats_plot.sum() * 100).values
x_pos = range(len(lane_stats_plot))

bars = ax10_main.bar(x_pos[:30], lane_stats_plot.values[:30], color='steelblue', alpha=0.7)
line = ax10_twin.plot(x_pos[:30], cumulative_pct[:30], color='red', marker='o',
                      linewidth=2, markersize=4, label='Cumulative %')
ax10_twin.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Line')

ax10_main.set_xlabel('Lane Rank', fontsize=11, fontweight='bold')
ax10_main.set_ylabel('Shipment Count', fontsize=11, fontweight='bold', color='steelblue')
ax10_twin.set_ylabel('Cumulative Percentage', fontsize=11, fontweight='bold', color='red')
ax10_main.set_title('10. Lane Concentration (Pareto Analysis)', fontsize=13, fontweight='bold')
ax10_main.tick_params(axis='y', labelcolor='steelblue')
ax10_twin.tick_params(axis='y', labelcolor='red')
ax10_twin.legend()
ax10_main.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Graph 11: Shipment Weight Distribution Pattern
fig11 = plt.figure(figsize=(10, 6))
ax11 = fig11.add_subplot(111)
scatter = ax11.scatter(df['Weight'], df.groupby('Weight').cumcount(),
                       alpha=0.3, s=20, c=df['Weight'], cmap='viridis')
ax11.axvline(x=5250, color='red', linestyle='--', linewidth=2, label='FTL Threshold (5,250 kg)')
ax11.axvline(x=df['Weight'].median(), color='orange', linestyle='--', linewidth=2,
             label=f'Median ({df["Weight"].median():.0f} kg)')
ax11.set_xlabel('Shipment Weight (kg)', fontsize=11, fontweight='bold')
ax11.set_ylabel('Occurrence Index', fontsize=11, fontweight='bold')
ax11.set_title('11. Shipment Weight Distribution Pattern', fontsize=13, fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax11, label='Weight (kg)')
plt.tight_layout()

# Graph 12: Consolidation Potential by Top Lanes
fig12 = plt.figure(figsize=(10, 8))
ax12 = fig12.add_subplot(111)

top_lanes = df['Lane'].value_counts().head(15).index
consolidation_by_lane = []
for lane in top_lanes:
    lane_data = df[df['Lane'] == lane]
    same_day = lane_data.groupby(lane_data['ShipDate'].dt.date).size()
    avg_per_day = same_day.mean()
    max_per_day = same_day.max()
    consolidation_by_lane.append({
        'Lane': lane.replace('-', ' → '),
        'Avg/Day': avg_per_day,
        'Max/Day': max_per_day
    })

cons_df = pd.DataFrame(consolidation_by_lane)
x = np.arange(len(cons_df))
width = 0.35

bars1 = ax12.barh(x, cons_df['Avg/Day'], width, label='Avg Shipments/Day',
                  color='lightblue', alpha=0.8)
bars2 = ax12.barh(x + width, cons_df['Max/Day'], width, label='Max Shipments/Day',
                  color='darkblue', alpha=0.8)

ax12.set_yticks(x + width / 2)
ax12.set_yticklabels(cons_df['Lane'], fontsize=9)
ax12.set_xlabel('Shipments per Day', fontsize=11, fontweight='bold')
ax12.set_title('12. Consolidation Potential: Top 15 Lanes', fontsize=13, fontweight='bold')
ax12.legend()
ax12.invert_yaxis()
ax12.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

# Graph 13: Cost Optimization Opportunity by Shipment Size
fig13 = plt.figure(figsize=(10, 6))
ax13_main = fig13.add_subplot(111)
ax13_twin = ax13_main.twinx()

weight_ranges = ['0-250', '251-500', '501-1000', '1001-2000', '2001-5000', '>5000']
shipment_counts = [
    len(df[df['Weight'] <= 250]),
    len(df[(df['Weight'] > 250) & (df['Weight'] <= 500)]),
    len(df[(df['Weight'] > 500) & (df['Weight'] <= 1000)]),
    len(df[(df['Weight'] > 1000) & (df['Weight'] <= 2000)]),
    len(df[(df['Weight'] > 2000) & (df['Weight'] <= 5000)]),
    len(df[df['Weight'] > 5000])
]

cost_reduction_potential = [40, 35, 30, 25, 15, 5]  # Percentage

colors = ['#FF4444', '#FF8844', '#FFBB44', '#FFDD44', '#AADD44', '#44DD44']
x_pos = range(len(weight_ranges))

bars = ax13_main.bar(x_pos, shipment_counts, color=colors, alpha=0.7, edgecolor='black')
line = ax13_twin.plot(x_pos, cost_reduction_potential, color='darkred', marker='D',
                      linewidth=3, markersize=10, label='Cost Reduction Potential')

ax13_main.set_xlabel('Weight Range (kg)', fontsize=11, fontweight='bold')
ax13_main.set_ylabel('Number of Shipments', fontsize=11, fontweight='bold', color='black')
ax13_twin.set_ylabel('Estimated Cost Reduction Potential (%)', fontsize=11, fontweight='bold',
                     color='darkred')
ax13_main.set_title('13. Cost Optimization Opportunity by Shipment Size', fontsize=13, fontweight='bold')
ax13_main.set_xticks(x_pos)
ax13_main.set_xticklabels(weight_ranges, rotation=45, ha='right')
ax13_twin.tick_params(axis='y', labelcolor='darkred')
ax13_twin.legend(loc='upper right')
ax13_main.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, shipment_counts):
    height = bar.get_height()
    ax13_main.text(bar.get_x() + bar.get_width() / 2., height + 30,
                   f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

print("\n✓ All visualizations generated successfully in separate windows!")
print("✓ Close each window to proceed to the next graph")
plt.show()
