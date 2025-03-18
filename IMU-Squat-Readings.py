import numpy as np
#https://numpy.org/doc/2.1/reference/routines.array-creation.html
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft
from scipy import stats


# Ieliekam datus, iznemam galveni
path = "C:/Users/User/Desktop/IMUData22.csv"
data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=float)

# Panemam laika kolonu
timestamps = data[:, 0]

# Panemam kopejo rindu skaitu
timestamps1 = data.shape[0]
print(timestamps1)

# Īstās laika vienības
new_timestamps = np.arange(0, timestamps1 * 20, 20).reshape(-1, 1)
print(new_timestamps)


# 50Hz paraugu ņemšanas ātrus priekš sensoriem. Vajadzētu būt 20ms
timestamp_differences = np.diff(timestamps)
print(timestamp_differences)

print(stats.mode(timestamp_differences)) # moda ir 0.019999980926513672, ne gluži 20ms, bet ir tuvu
