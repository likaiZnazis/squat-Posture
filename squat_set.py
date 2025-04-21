import numpy as np
from dataclasses import dataclass
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
    segmentArray: np.ndarray
    rawData: np.ndarray
    segmentAmountBefore: int = 0
    segmentAmountAfter: int = 0

    # Method that will fill out the variables, like segment amount
    def trimSegments(self) -> None:
        # Assume that the first and last segment are faulty
        if(len(self.segmentArray) == 12):
            self.segmentAmountBefore = len(self.segmentArray)
            self.segmentArray = self.segmentArray[1:len(self.segmentArray)-1]
            self.segmentAmountAfter = len(self.segmentArray)

# currentFile = np.loadtxt(os.path.join(os.getcwd(),"dataset","1-Set.csv"),dtype="float", skiprows=1, delimiter=",")
# segmentedFile = preprocessing.get_segment_indexes(currentFile)
# set1 = Set("Set-1", "Wide_knees", segmentedFile)
# set1.getSegmentAmount()
# print(set1)