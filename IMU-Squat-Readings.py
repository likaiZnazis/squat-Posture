# import numpy as np

# # Load the header separately
# with open("C:/Users/User/Desktop/IMUData.csv", "r") as f:
#     header = f.readline().strip().split(",")

# # Load the data (skip the first row)
# data = np.loadtxt("C:/Users/User/Desktop/IMUData.csv", skiprows=1, delimiter=",", dtype=float)

# # Get the timestamps. First column
# timestamps = data.shape[0]
# new_timestamps = np.arange(0, timestamps * 20, 20).reshape(-1, 1)  
# # Used 50Hz to read from sensors ~ each difference should be around 20ms 
# # timestamp_differences = np.diff(timestamps)# more or less around 20ms
# # print(timestamp_differences)
# values = data[:, 1:]
# print(new_timestamps)

# formatted_data = np.column_stack((new_timestamps, values))
# print(formatted_data)
# # Print results
# # print(",".join(header))  # Print the header
# # print(data[0])  # Print the normalized data

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the header separately
with open("C:/Users/User/Desktop/IMUData.csv", "r") as f:
    header = f.readline().strip().split(",")

# Load the data (skip the first row)
data = np.loadtxt("C:/Users/User/Desktop/IMUData.csv", skiprows=1, delimiter=",", dtype=float)

# Generate new timestamps (assuming 50Hz, so 20ms intervals)
timestamps = np.arange(0, data.shape[0] * 20, 20)  # In milliseconds

# Extract sensor values (excluding timestamp column)
sensor_values = data[:, 1:]  # All columns except the first

# Plot each sensor variable
plt.figure(figsize=(12, 6))
for i in range(sensor_values.shape[1]):  # Loop through each sensor column
    plt.plot(timestamps, sensor_values[:, i], label=header[i + 1])  # Use header names

plt.xlabel("Time (ms)")
plt.ylabel("Sensor Values")
plt.title("IMU Sensor Data Over Time")
plt.legend()
plt.grid()
plt.show()
