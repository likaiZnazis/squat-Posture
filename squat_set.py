import numpy as np
from dataclasses import dataclass, field
from scipy import stats
import os
import preprocessing
"""
Vajadzētu uztaisīt klases, kur ir sets un ir dataset, kas sastāv no vairākiem set.
Klases tiks izveidotas, lai būtu vieglāk aplūkot inormāciju.

"""
@dataclass
class SquatSet:
    fileName: str
    formName: str
    segmentArray: list[int] = field(default_factory=list)
    finalArray: list[int] = field(default_factory=list)
    rawData: np.ndarray
    segmentAmountBefore: int = 0
    segmentAmountAfter: int = 0

    SENSOR_MEASURMENTS = {
        "time": 0,
        "accX": 1,
        "accY": 2,
        "accZ": 3,
        "gyroX": 4,
        "gyroY": 5,
        "gyroZ": 6,
        "magX": 7,
        "magY": 8,
        "magZ": 9,
        "pitch": 10,
        "roll": 11,
        "yaw": 12,
    }

    # Method that will fill out the variables, like segment amount
    def trimSegments(self) -> None:
        # Assume that the first and last segment are faulty
        if(len(self.segmentArray) == 12):
            self.segmentAmountBefore = len(self.segmentArray)
            self.segmentArray = self.segmentArray[1:len(self.segmentArray)-1]
            self.segmentAmountAfter = len(self.segmentArray)
    
    #source - #https://github.com/mlgig/Video_vs_Shimmer_ECML_2023/blob/master/utils/math_funtions.py
    def get_segment_indexes(self):
        #Creates a array with starting and ending indexes for each segment
        segment_signal = self.rawData[:, self.SENSOR_MEASURMENTS["pitch"]]
        threshold = (np.mean(segment_signal) + np.std(segment_signal)) - 55
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
        self.segmentArray = final_pairs
    
    def extract_each_signal(self):
        #Using starting and edning indexes extracts each sensor signal from the original dataset. Resulting in individual segments
        segmented_squats = []
        for start,end in self.segmentArray:
            # Iznemam visas vajadzīgās vērtības
            segment_data = self.rawData[start:end + 1, [self.SENSOR_MEASURMENTS[sensor_signal] for sensor_signal in self.SENSOR_MEASURMENTS]]
            segmented_squats.append(segment_data)
        return segmented_squats
    
    def getFrequency(self,dataset):
        #WE load the dataset
        data = np.loadtxt(dataset,dtype="float", skiprows=1, delimiter=",")
        #Grab timestamps
        timestamps = data[:,0]
        #Get differences between each point
        #For example 1743403873.08 - 1743403873.10 = 
        timeStampDifference = np.diff(timestamps)
        #Get common value
        return stats.mode(timeStampDifference)

# currentFile = np.loadtxt(os.path.join(os.getcwd(),"dataset","1-Set.csv"),dtype="float", skiprows=1, delimiter=",")
# segmentedFile = preprocessing.get_segment_indexes(currentFile)
# set1 = Set("Set-1", "Wide_knees", segmentedFile)
# set1.getSegmentAmount()
# print(set1)