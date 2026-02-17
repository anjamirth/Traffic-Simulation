import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
data = scipy.io.loadmat('traffic_dataset.mat')
tra_Y_tr = data['tra_Y_tr']  # Shape: (36 locations, 1261 time steps)

num_locations, num_time_steps = tra_Y_tr.shape

# 2. Create the plot
plt.figure(figsize=(15, 8))

# Generate 36 distinct colors from a colormap
colors = plt.cm.jet(np.linspace(0, 1, num_locations))

# 3. Plot each location
for i in range(num_locations):
    plt.plot(tra_Y_tr[i, :], 
             color=colors[i], 
             alpha=0.6, 
             linewidth=0.8, 
             label=f'Loc {i}')

plt.title("Traffic Flow: All 36 Locations (Jan 2017)")
plt.xlabel("Time (Quarter-Hour Intervals)")
plt.ylabel("Traffic Flow")

# 4. Refine the Legend
# Since 36 items is a lot, we can display it in multiple columns
plt.legend(loc='upper right', ncol=3, fontsize='x-small', title="Sensor Locations")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- Optional: Zoomed view for the first 3 days (96 intervals * 3) ---
plt.figure(figsize=(15, 6))
for i in range(num_locations):
    plt.plot(tra_Y_tr[i, :288], color=colors[i], alpha=0.7, linewidth=1)

plt.title("Traffic Flow: All Locations (First 72 Hours)")
plt.xlabel("Time (Quarter-Hour Intervals)")
plt.ylabel("Traffic Flow")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()