import numpy as np
#https://numpy.org/doc/2.1/reference/routines.array-creation.html
import matplotlib.pyplot as plt
import os
#https://matplotlib.org/stable/plot_types/index.html

#Signala apstradasana, pievienoja filtru shoutouts AI
# from scipy.signal import butter, filtfilt
# from scipy.fft import fft, ifft

#importējam klasifikācijas modeli
# from aeon.classification.convolution_based import MultiRocketHydraClassifier

# from scipy import stats
# Prieks modas, medianas un citam statiem

# Ieliekam datus, iznemam galveni
# path = "C:/Users/User/Desktop/Bakalaurs/Mans BD/Programmesana/Dataset/dataset/IMUData22.csv"
# data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=float)

# 50Hz paraugu ņemšanas ātrus priekš sensoriem. Vajadzētu būt 20ms
# timestamp_differences = np.diff(timestamps)
# print(timestamp_differences)

# print(stats.mode(timestamp_differences)) # moda ir 0.019999980926513672, ne gluži 20ms, bet ir tuvu

fs = 50  # Iestatītā paraugu ņemšanas frekvence
# new_timestamps = np.arange(0, timestamps1 * 20, 20)

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
}

#Segmenta metodi panemu no cita pētījuma
#https://github.com/mlgig/Video_vs_Shimmer_ECML_2023/blob/master/utils/math_funtions.py

# ---- 3. Segment Detection ----
def get_segment_indexes(file):
    #Creates a array with starting and ending indexes for each segment
    segment_signal = file[:, sensor_measurments["pitch"]]
    threshold = (np.mean(segment_signal) + np.std(segment_signal)) - 25
    marker_list = [i >= threshold for i in segment_signal]
    i = 0
    final_pairs = []
    
    while i < len(segment_signal):
        if marker_list[i]:  # Start of a segment
            start = i
            while i < len(segment_signal) and marker_list[i]:
                i += 1
            end = i - 1  # End of the segment
            if end - start > 1:  # Ensure segment is significant
                final_pairs.append((start, end))
        i += 1
    return final_pairs


# ---- 3. Iegūstam segmentus ----
# segments = get_segments(segment_signal, threshold)
# Segmentus vajadzētu likt citā rindā
#https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MultiRocketHydraClassifier.html#aeon.classification.convolution_based.MultiRocketHydraClassifier.fit
# Si ir datu rinda kadu MultiRocketHydraClassifier pienem
# make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12) - (segmenti, iezimes, merijumu paraugi)

#Test for right amount of segments = 10 test for returned segments
"""
Problem with this function is that each segment will be different type of shape.
Meaning that if I do a long squat there will be more measurments than in a short one.
np.array() does not allow that. Setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.
Can do it only in standart array.
"""
def extract_each_signal(segment_index_array, data):
    #Using starting and edning indexes extracts each sensor signal from the original dataset. Resulting in individual segments
    segmented_squats = []
    for start,end in segment_index_array:
        # Iznemam visas vajadzīgās vērtības
        segment_data = data[start:end + 1, [sensor_measurments[sensor_signal] for sensor_signal in sensor_measurments]]
        segmented_squats.append(segment_data)
    return segmented_squats[1:11]#2D array

def resample_segments(segmented_squats):
    longest_squat_measured = max(len(segment) for segment in segmented_squats)
    segmented_resampled_squats = np.array([np.pad(segment, ((0, longest_squat_measured - len(segment)), (0, 0)), mode='constant') for segment in segmented_squats])
    segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
    return segmented_resampled_squats
"""
# for squat in segmented_squats:
#     # Izprintejam pietupienu mērījumu punktu skaitu
#     print(len(squat))

# # Izmeram cik katra pietupiena segmenta ir paraugu merijumi
# # squat_segment_samples = [len(segment) for segment in segmented_squats]#izmantojam list comprehension
# # print(squat_segment_samples)

# # Iegūstam garāko pietupiena izpildījumu

# # print(longest_squat_measured)

# # Atkārtoti paraugojam pietupiena segmentus. np.pad(array, {sequence, array_like, int}, mode dazadi). Iespējams otru argumentu var pamainīt
# 'constant' - atkāti paraugo ar konstantu vērtību

# for squat in segmented_resampled_squats:
#     # Izprintejam pietupienu mērījumu punktu skaitu
#     print(len(squat))

# print(segmented_resampled_squats.shape)#(10, 41, 10)
# #3D datu rinda: 10 segmenti, 41 iezimes, 10 merijumu paraugi
# #Sis nav korekti, jo mums ir 10 segmenti, 10 iezimes, 41 merijumu paraugi

# #Apmainam iezimes ar merijumu paraugiem
# segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
# print(segmented_resampled_squats.shape)#(10, 10, 41)

#Iespejamās klases, atspogulotas ar cipariem, modelim vajadzigi cipari, kas atspogulo klases
# label_to_number = {
#     "bad-squat": 0,
#     "good-squat": 1,
# }

#Veikto pietupienu markejums, vajag but vienadam ar segmentu daudzumu
# squat_labels = ["bad-squat", "bad-squat", "bad-squat", "bad-squat", "good-squat", "good-squat", "good-squat", "bad-squat", "good-squat", "good-squat"]

# Markejam datus ar klasem
# segment_labels = np.array([label_to_number[label] for label in squat_labels])
# print(segment_labels)

# ---- Modeļa treniņs
# Izveidojam modeļa instanci
# clf = MultiRocketHydraClassifier(random_state=0)

# Liekam segmentetos pietupienus ar markejumiem, katrs segments atbilst 1D rindas, kur ieksa ir markejumi
# clf.fit(segmented_resampled_squats, segment_labels)

# Parbaudam ka modelis klasifice pietupienus
# print(clf.predict(segmented_resampled_squats))
"""

# ---- 4. Attēlot datus ar atpazītajiem segmentiem ----
def show_graph(file):
    data = np.loadtxt(file, delimiter=",",dtype="float",skiprows=1)
    sensor_values = data[:,sensor_measurments["pitch"]]
    new_timestamps = np.arange(0, data.shape[0] * 20, 20)
    segments = get_segment_indexes(data)
    plt.figure(figsize=(12, 6))  # Izveidojam grafiku ar izmēru 12x6 collas

    # Uzzīmējam sākotnējos sensora mērījumus, alpha ir krasas caurspidigums
    plt.plot(new_timestamps, sensor_values, label='Garneslīpuma mērījumi', color='red', alpha=0.3)

    # Marķējam atpazītos segmentus grafikā
    for (start, end) in segments:
        plt.axvspan(new_timestamps[start], new_timestamps[end], color='yellow',  alpha=0.3, 
                    label='Pietupiena segments' if start == segments[0][0] else "")  #Leģendu pievienojam tikai vienu reizi, caur for loop ta ir attiecigas reizes vairak

    # Noformējam grafika aprakstus
    plt.xlabel("Laiks (ms)" , fontsize=18)  # X ass nosaukums
    plt.ylabel("Garenslīpuma mērījumi (°)", fontsize=18)  # Y ass nosaukums
    plt.title("Sensoru mērījumu vizualizācija", fontsize=18)  # Grafika galvenes nosaukums
    plt.legend()  # Pievienojam leģendu
    plt.grid()  # Pievienojam režģi
    plt.show()  # Parādam grafiku'

file = os.path.join(os.getcwd(),"dataset","test_dataset.npy")

# print(file)
# show_graph(file)
data = np.load(file)
# print(data.shape)
print(data)
# np.save(os.path.join(os.path.join(os.getcwd(),"dataset"), "test"), np.array([1,2,3,4]))
