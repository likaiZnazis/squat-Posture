import numpy as np
#https://numpy.org/doc/2.1/reference/routines.array-creation.html
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
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
        segment_data = data[start:end + 1, [sensor_measurments[sensor_signal] for sensor_signal in sensor_measurments]]
        segmented_squats.append(segment_data)
    if (len(segmented_squats) == 12):
        return segmented_squats[1:11]
    else:
        return segmented_squats
    #2D array

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

#Make that this function takes in a single squat
def make_graph(file):
    data = np.loadtxt(file, delimiter=",", dtype="float", skiprows=1)

    plt.figure(figsize=(12, 8))

    # 1. AccX over Time
    plt.subplot(2, 3, 1)
    plt.scatter(data[:, sensor_measurments["time"]],
                data[:, sensor_measurments["accX"]],
                color='red', s=5)
    plt.title("AccX over Time")
    plt.xlabel("Time")
    plt.ylabel("AccX")

    # 2. Pitch over Time
    plt.subplot(2, 3, 2)
    plt.scatter(data[:, sensor_measurments["time"]],
                data[:, sensor_measurments["pitch"]],
                color='blue', s=5)
    plt.title("Pitch over Time")
    plt.xlabel("Time")
    plt.ylabel("Pitch")

    # 3. AccX vs GyroX
    plt.subplot(2, 3, 3)
    plt.scatter(data[:, sensor_measurments["accX"]],
                data[:, sensor_measurments["gyroX"]],
                color='green', s=5)
    plt.title("AccX vs GyroX")
    plt.xlabel("AccX")
    plt.ylabel("GyroX")

    # 4. Pitch vs Roll
    plt.subplot(2, 3, 4)
    plt.scatter(data[:, sensor_measurments["pitch"]],
                data[:, sensor_measurments["roll"]],
                color='orange', s=5)
    plt.title("Pitch vs Roll")
    plt.xlabel("Pitch")
    plt.ylabel("Roll")

    # 5. AccZ vs GyroZ
    plt.subplot(2, 3, 5)
    plt.scatter(data[:, sensor_measurments["accZ"]],
                data[:, sensor_measurments["gyroZ"]],
                color='purple', s=5)
    plt.title("AccZ vs GyroZ")
    plt.xlabel("AccZ")
    plt.ylabel("GyroZ")

    plt.tight_layout()
    plt.show()

# file = os.path.join(os.getcwd(),"dataset","7-Half-Set.csv")
# make_graph(file)

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

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def compare_acc_xyz_two_experiments(file1, file2, sensor):
    if sensor not in ["acc", "gyro", "other"]:
        print("Invalid sensor type. Use 'acc', 'gyro', or 'other'.")
        return

    sensor_measurments = {
        "accX": 1, "accY": 2, "accZ": 3,
        "gyroX": 4, "gyroY": 5, "gyroZ": 6,
        "pitch": 7, "roll": 8, "yaw": 9
    }

    # Load data
    data1 = np.loadtxt(file1, delimiter=",", dtype="float", skiprows=1)
    data2 = np.loadtxt(file2, delimiter=",", dtype="float", skiprows=1)

    # Get sensor values
    if sensor in ["acc", "gyro"]:
        s1 = [data1[:, sensor_measurments[sensor + axis]] for axis in ["X", "Y", "Z"]]
        s2 = [data2[:, sensor_measurments[sensor + axis]] for axis in ["X", "Y", "Z"]]
    else:
        s1 = [data1[:, sensor_measurments[axis]] for axis in ["pitch", "roll", "yaw"]]
        s2 = [data2[:, sensor_measurments[axis]] for axis in ["pitch", "roll", "yaw"]]

    # Timestamps
    time1 = np.arange(0, len(s1[0]) * 20, 20)
    time2 = np.arange(0, len(s2[0]) * 20, 20)

    # Setup plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Style settings
    axis_labels = ["X", "Y", "Z"]
    line_style = "-"
    colors = ["blue", "orange"]  # Exp 1 = blue, Exp 2 = orange
    sensors = ["garenslīpums", "sānsverpe", "virsgrieze"]

    for i in range(3):
        axs[i].plot(time1, s1[i], color=colors[0], linestyle=line_style, label=f"Exp 1 - {axis_labels[i]}")
        axs[i].plot(time2, s2[i], color=colors[1], linestyle=line_style, label=f"Exp 2 - {axis_labels[i]}")
        axs[i].set_ylabel(f"{sensors[i].upper()} °")
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel("Laiks (ms)")
    plt.suptitle("Rotāciju leņķu salīdzināšana", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def getFrequency():
    #WE load the dataset
    data = np.loadtxt(os.path.join(os.getcwd(),"dataset","1-Full-Set.csv"),dtype="float",delimiter=",",skiprows=1)
    #Grab timestamps
    timestamps = data[:,0]
    #Get differences between each point
    
    #Calculate difference between 2 timepoints
    timeStampDifference = np.diff(timestamps)
    #Count how many times a
    values, counts = np.unique(timeStampDifference, return_counts=True)
    frequency = 1 / values[np.argmax(counts)]
    frequency = (math.ceil(frequency))
    print(frequency)
# getFrequency()
# file1 = os.path.join(os.getcwd(),"1expe.csv")
# file2 = os.path.join(os.getcwd(),"2expe.csv")

# compare_acc_xyz_two_experiments(file1=file1, file2=file2, sensor="other")

data1 = os.path.join(os.getcwd(),"dataset","test_dataset_labels.npy")
# label = os.path.join(os.getcwd(),"dataset","test_dataset_labels.npy")
# file = os.path.join(os.getcwd(),"dataset","2-Full-Set.csv")
# data = np.load(data1)
# print(data)
# show_graph(file1)
# dataset_data = np.load(data)

# plot_squat_class_distribution(dataset_data,dataset_labels)
