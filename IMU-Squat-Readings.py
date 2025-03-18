import numpy as np
#https://numpy.org/doc/2.1/reference/routines.array-creation.html
import matplotlib.pyplot as plt
#https://matplotlib.org/stable/plot_types/index.html

#Signala apstradasana, pievienoja filtru shoutouts AI
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft

#importējam klasifikācijas modeli
from aeon.classification.convolution_based import MultiRocketHydraClassifier

# from scipy import stats
# Prieks modas, medianas un citam statiem

# Ieliekam datus, iznemam galveni
path = "C:/Users/maris/Downloads/IMUData22.csv"
data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=float)

# Panemam laika kolonu
timestamps = data[:, 0]

# Panemam kopejo rindu skaitu
timestamps1 = data.shape[0]
# print(timestamps1)

# Īstās laika vienības

# print(new_timestamps)


# 50Hz paraugu ņemšanas ātrus priekš sensoriem. Vajadzētu būt 20ms
# timestamp_differences = np.diff(timestamps)
# print(timestamp_differences)

# print(stats.mode(timestamp_differences)) # moda ir 0.019999980926513672, ne gluži 20ms, bet ir tuvu

fs = 50  # Iestatītā paraugu ņemšanas frekvence
new_timestamps = np.arange(0, timestamps1 * 20, 20)

# Sensora CSV mainīgo nosaukumi un attiecīgā kollona
sensor_measurments = {
    "time": 0,
    "accX": 1,
    "accY": 2,
    "accZ": 3,
    "gyroX": 4,
    "gyroY": 5,
    "gyroZ": 6,
    "pitch": 7,
    "roll": 8,
    "yaw": 9,
    # Altitude bija izmeginājums, domāju iespējams būs labs mērījums priekš segmentācijas
    # Ļoti jocīgi mēra, krīt palēnām uz leju, diezgan bezjēdzīgs
}

# Sensora kolona, kas tiks atspoguļota grafikā atkarībā no laika
sensor_values = data[:, sensor_measurments["pitch"]]  
# print(sensor_values)

#AI palidzēja ar filtru, lidz galam nesaprotu kas notiek
# ---- 1. Apply Butterworth Low-Pass Filter (fc = 20 Hz, order = 8) ----
def butter_lowpass(cutoff, fs, order=8):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the filter
cutoff_freq = 20  # Hz
b, a = butter_lowpass(cutoff_freq, fs)
butter_filtered_data = filtfilt(b, a, sensor_values)

# ---- 2. Apply Fourier Transform Filtering ----
signal_fft = fft(butter_filtered_data)  # Compute FFT after Butterworth filtering
n = len(butter_filtered_data)
frequencies = np.fft.fftfreq(n, d=(1/fs))

# Remove frequencies above cutoff
fourier_cutoff = 20  # Hz
cutoff_index = np.where(frequencies > fourier_cutoff)[0][0]
signal_fft[cutoff_index:] = 0  # Zero out high frequencies

# Convert back to time domain
fourier_filtered_data = np.real(ifft(signal_fft))

#Segmenta metodi panemu no cita pētījuma
#https://github.com/mlgig/Video_vs_Shimmer_ECML_2023/blob/master/utils/math_funtions.py
# ---- 3. Segment Detection ----
def get_segments(weights, threshold):
    """
    Identify continuous segments where values exceed the threshold.
    """
    marker_list = [i >= threshold for i in weights]
    i = 0
    final_pairs = []
    
    while i < len(weights):
        if marker_list[i]:  # Start of a segment
            start = i
            while i < len(weights) and marker_list[i]:
                i += 1
            end = i - 1  # End of the segment
            if end - start > 1:  # Ensure segment is significant
                final_pairs.append((start, end))
        i += 1
    return np.array(final_pairs)

# Set threshold dynamically based on Fourier filtered data
threshold = np.mean(fourier_filtered_data) + np.std(fourier_filtered_data)


segments = get_segments(fourier_filtered_data, threshold)
# Segmentus vajadzētu likt citā rindā
#https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MultiRocketHydraClassifier.html#aeon.classification.convolution_based.MultiRocketHydraClassifier.fit
# Si ir datu rinda kadu MultiRocketHydraClassifier pienem
# make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12) - (segmenti, iezimes, merijumu paraugi)

segmented_squats = []

for start, end in segments:
    # Izgriezam rindas. Tas ir rindas, garumi sakrīt
    # print(len(data[start:end+1]))
    # Iznemam visas vajadzīgās vērtības
    segment_data = data[start:end + 1, [sensor_measurments[sensor_signal] for sensor_signal in sensor_measurments]]  
    segmented_squats.append(segment_data)


# Izgriezam pirmo un pedejo pietupienu, jo tie nav pietupieni
segmented_squats = segmented_squats[1:11]

# for squat in segmented_squats:
#     # Izprintejam pietupienu mērījumu punktu skaitu
#     print(len(squat))

# # Izmeram cik katra pietupiena segmenta ir paraugu merijumi
# # squat_segment_samples = [len(segment) for segment in segmented_squats]#izmantojam list comprehension
# # print(squat_segment_samples)

# # Iegūstam garāko pietupiena izpildījumu
longest_squat_measured = max(len(segment) for segment in segmented_squats)
# # print(longest_squat_measured)

# # Atkārtoti paraugojam pietupiena segmentus. np.pad(array, {sequence, array_like, int}, mode dazadi). Iespējams otru argumentu var pamainīt
segmented_resampled_squats = np.array([np.pad(segment, ((0, longest_squat_measured - len(segment)), (0, 0)), mode='constant') for segment in segmented_squats])# 'constant' - atkāti paraugo ar konstantu vērtību

# for squat in segmented_resampled_squats:
#     # Izprintejam pietupienu mērījumu punktu skaitu
#     print(len(squat))

# print(segmented_resampled_squats.shape)#(10, 41, 10)
# #3D datu rinda: 10 segmenti, 41 iezimes, 10 merijumu paraugi
# #Sis nav korekti, jo mums ir 10 segmenti, 10 iezimes, 41 merijumu paraugi

# #Apmainam iezimes ar merijumu paraugiem
segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
# print(segmented_resampled_squats.shape)#(10, 10, 41)

#Iespejamās klases, atspogulotas ar cipariem, modelim vajadzigi cipari, kas atspogulo klases
label_to_number = {
    "bad-squat": 0,
    "good-squat": 1,
}

#Veikto pietupienu markejums, vajag but vienadam ar segmentu daudzumu
squat_labels = ["bad-squat", "bad-squat", "bad-squat", "bad-squat", "good-squat", "good-squat", "good-squat", "bad-squat", "good-squat", "good-squat"]

# Markejam datus ar klasem
segment_labels = np.array([label_to_number[label] for label in squat_labels])
# print(segment_labels)

# ---- Modeļa treniņs
# Izveidojam modeļa instanci
clf = MultiRocketHydraClassifier(random_state=0)

# Liekam segmentetos pietupienus ar markejumiem, katrs segments atbilst 1D rindas, kur ieksa ir markejumi
clf.fit(segmented_resampled_squats, segment_labels)

# Parbaudam ka modelis klasifice pietupienus
print(clf.predict(segmented_resampled_squats))

# ---- 4. Attēlot datus ar atpazītajiem segmentiem ----
'''
plt.figure(figsize=(12, 6))  # Izveidojam grafiku ar izmēru 12x6 collas

# Uzzīmējam sākotnējos sensora mērījumus, alpha ir krasas caurspidigums
plt.plot(new_timestamps, sensor_values, label='Orģinālie mērījumi', color='red', alpha=0.3)

# Uzzīmējam Butterworth zemo frekvenču filtru datus (var atkomentēt, ja nepieciešams)
# plt.plot(new_timestamps, butter_filtered_data, label='Butterworth Filtered Data (20 Hz, Order 8)', color='green', alpha=0.6)

# Marķējam atpazītos segmentus grafikā
for (start, end) in segments:
    plt.axvspan(new_timestamps[start], new_timestamps[end], color='yellow', alpha=0.3, 
                label='Pietupiena segments' if start == segments[0][0] else "")  #Leģendu pievienojam tikai vienu reizi, caur for loop ta ir attiecigas reizes vairak

# Uzzīmējam Fourier transformācijas filtrētos datus
plt.plot(new_timestamps, fourier_filtered_data, label='Fourier Transform filtrēti mērījumi', color='blue', linewidth=2)

# Noformējam grafika aprakstus
plt.xlabel("Laiks (ms)")  # X ass nosaukums
plt.ylabel("Sensora vērtības mērījumi")  # Y ass nosaukums
plt.title("Garenslīpuma mērījumi")  # Grafika galvenes nosaukums
plt.legend()  # Pievienojam leģendu
plt.grid()  # Pievienojam režģi
plt.show()  # Parādam grafiku'
'''
