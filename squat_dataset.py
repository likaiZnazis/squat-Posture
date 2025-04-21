from dataclasses import dataclass, field
from squat_set import SquatSet
import numpy as np
import traceback
import preprocessing
import os

@dataclass
class Dataset:
    allSets: list[SquatSet] = field(default_factory=list)
    trainingSet: list[SquatSet] = field(default_factory=list)
    testingSet: list[SquatSet] = field(default_factory=list)
    #Path to both files
    trainingSetPath: str
    testingSetPath: str

    def resample_segments(self):
        longest_squat_measured = max(len(segment) for segment in self.segmentArray)
        segmented_resampled_squats = np.array([np.pad(segment, ((0, longest_squat_measured - len(segment)), (0, 0)), mode='constant') for segment in self.segmentArray])
        segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
        return segmented_resampled_squats

    def combine_sets():
        try:
            # Get path to the set files. They will start with a number
            pathToSetFiles = [ os.path.join(path, file) for file in os.listdir(path) if (file[0].isdigit())]
            # Get segments from each file
            all_signals = []
            for pathSetFile in pathToSetFiles:
                #load the file
                currentFile = np.loadtxt(pathSetFile,dtype="float", skiprows=1, delimiter=",")
                #Get segment indexes
                segment_indexes = preprocessing.get_segment_indexes(currentFile)
                #Extract each sensor signal from the segment indexes
                sensor_signals = preprocessing.extract_each_signal(segment_indexes, currentFile) 
                #sensor_signals is a 2D array
                for row in sensor_signals:
                    all_signals.append(row)
            return all_signals

        except OSError as err:
            print("OS error occured " + "probably the path is not correct" + " " + str(err))
            print(traceback.format_exc())
        except Exception as err:
            print("Dont know how to handle this error " + str(err))
            print(traceback.format_exc())
            # raise - reraising the error is only useful if there are higher-up try,except
        except ValueError:
            print("There was a value error " + str(err))
            print(traceback.format_exc())

    def split_sets():
        pass
