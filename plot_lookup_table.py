import csv
import numpy as np
import matplotlib.pyplot as plt
import os

csv_path = os.path.join(os.path.dirname(__file__) or '.', 'lookup_table.csv')

cmap='viridis'

# Read CSV skipping comment lines
rows = []
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        if row[0].strip().startswith('#'):
            continue
        rows.append(row)

if not rows or len(rows) < 2:
    raise SystemExit("CSV doesn't contain expected header + data rows")

header = rows[0]
velocities = np.array([float(v) for v in header[1:]], dtype=float)

heights_list = []
angle_rows = []
for row in rows[1:]:
    h = float(row[0])
    heights_list.append(h)
    row_angles = []
    for cell in row[1:]:
        if cell is None or cell.strip() == "":
            row_angles.append(np.nan)
        else:
            parts = cell.split(';')
            try:
                angle = float(parts[0])
            except Exception:
                angle = np.nan
            row_angles.append(angle)
    angle_rows.append(row_angles)

heights = np.array(heights_list, dtype=float)
angles = np.array(angle_rows, dtype=float)  # shape (len(heights), len(velocities))

# Prepare edges for pcolormesh
if velocities.size == 1:
    v_edges = np.array([velocities[0] - 0.5, velocities[0] + 0.5])
else:
    v_diffs = np.diff(velocities) / 2.0
    v_edges = np.empty(velocities.size + 1, dtype=float)
    v_edges[1:-1] = velocities[:-1] + v_diffs
    v_edges[0] = velocities[0] - v_diffs[0]
    v_edges[-1] = velocities[-1] + v_diffs[-1]

if heights.size == 1:
    h_edges = np.array([heights[0] - 0.5, heights[0] + 0.5])
else:
    h_diffs = np.diff(heights) / 2.0
    h_edges = np.empty(heights.size + 1, dtype=float)
    h_edges[1:-1] = heights[:-1] + h_diffs
    h_edges[0] = heights[0] - h_diffs[0]
    h_edges[-1] = heights[-1] + h_diffs[-1]

angles_masked = np.ma.masked_invalid(angles)

fig, ax = plt.subplots(figsize=(8,6))
cmap_obj = plt.get_cmap(cmap)
cmap_obj.set_bad(color='lightgray')

pcm = ax.pcolormesh(v_edges, h_edges, angles_masked, cmap=cmap_obj, shading='auto')
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Deployment angle (deg)')

ax.set_xlabel('Burnout velocity (m/s)')
ax.set_ylabel('Burnout height (m)')
ax.set_title('Lookup table: deployment angle (deg)')

plt.tight_layout()
plt.show()
plt.close(fig)