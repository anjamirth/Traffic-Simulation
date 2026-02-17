import matplotlib.pyplot as plt
import numpy as np
import scipy.io
data = scipy.io.loadmat('traffic_dataset.mat')

train_y = data['tra_Y_tr']

def display_location(location_index):

    location_flow = train_y[location_index, :]
    time_steps = np.arange(len(location_flow))

    plt.figure(figsize=(15, 6))
    plt.plot(time_steps, location_flow, color='red', linewidth=1)

    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.grid(False, linestyle='--', alpha=0.7)

    
    # plt.xlim(0, 96*7)
    plt.legend([f'Location {location_index}'])
    plt.show()

display_location(35)