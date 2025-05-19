from dataclasses import dataclass, field
from squat_set import SquatSet
import numpy as np
import traceback
import preprocessing
import os
import math
from scipy import stats

@dataclass
class Dataset:
    allSets: list[SquatSet] = field(default_factory=list)
    trainingSet: list[SquatSet] = field(default_factory=list)
    testingSet: list[SquatSet] = field(default_factory=list)
    #Path to both files
    trainingSetPath: str = field(default_factory=str)
    testingSetPath: str= ""
    path: str = os.path.join(os.getcwd(), "dataset")
    frequency: int = field(default_factory=int)

    def getFrequency(self) -> int:
        #WE load the dataset
        print("path "+ self.trainingSetPath)
        data = np.load(self.trainingSetPath)
        #Grab timestamps
        timestamps = data[:,0]
        #Get differences between each point
        
        #Calculate difference between 2 timepoints
        timeStampDifference = np.diff(timestamps)

        #Count how many times a
        values, counts = np.unique(timeStampDifference, return_counts=True)
        self.frequency = 1 / values[np.argmax(counts)]
        self.frequency = (math.ceil(self.frequency))
        return self.frequency
    
    def combine_all_sets(self) -> int:
        try:
            # Get path to the set files. They will start with a number
            # print(os.listdir(self.path))
            files = [f for f in os.listdir(self.path) if f[0].isdigit()]
            files = sorted(files, key=lambda f: int(f.split('-')[0]))
            total_reps = 0
            # Get segments from each file
            for setFileName in files:
                # print(setFileName.split("\\")[-1])
                newSet = SquatSet(fileName=setFileName.split("\\")[-1])
                #load the file
                newSet.formName = setFileName.split("\\")[-1].split("-")[1]
                print(setFileName)
                currentFile = np.loadtxt(os.path.join(self.path,setFileName),dtype="float", skiprows=1, delimiter=",")
                #Get segment indexes
                segment_indexes = preprocessing.get_segment_indexes(currentFile)
                newSet.segmentAmountBefore = len(segment_indexes)
                #Extract each sensor signal from the segment indexes
                newSet.each_sginal_segmented = preprocessing.extract_each_signal(segment_indexes, currentFile)
                newSet.segmentAmountAfter = len(newSet.each_sginal_segmented)
                total_reps+=newSet.segmentAmountAfter
                self.allSets.append(newSet)
                
            return total_reps
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

    #Splits all the sets into training and testing sets
    def split_sets(self) -> None:
        seen_forms = set()
        for squatSet in self.allSets:
            if(squatSet.formName not in seen_forms):
                #Add the form
                self.testingSet.append(squatSet)
                seen_forms.add(squatSet.formName)
            else:
                self.trainingSet.append(squatSet)

        form_count = {}
        for squatSet in self.allSets:
            print(squatSet.formName)
            if squatSet.formName in form_count:
                form_count[squatSet.formName] += 10
            else:
                form_count[squatSet.formName] = 10
        return form_count
    
    def longest_squatset_segment(self) -> int:
        #this will length the object
        return max(len(squatSet) for squatSet in self.allSets)

    #From both lists create files
    def train_file(self):
        #For each training set resample it and add it to a numpy array
        train_dataset = []
        labels = []
        #Dont know if this is the best way to add the dataset
        for squatSet in self.trainingSet:
            labels.append(squatSet.formName)
            #resample all segments to equal size
            segmented_resampled_squats = np.array([np.pad(segment, ((0, self.longest_squatset_segment() - len(segment)), (0, 0)), mode='constant') for segment in squatSet.each_sginal_segmented])
            segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
            train_dataset.append(segmented_resampled_squats)

        train_dataset = np.concatenate(train_dataset, axis=0)
        train_dataset_labels = np.array([label for label in labels for _ in range(10)])
        #Check if the path exists
        np.save(os.path.join(self.path, "train_dataset_labels"), train_dataset_labels)
        np.save(os.path.join(self.path, "train_dataset"), train_dataset)
        self.trainingSetPath = os.path.join(self.path, "train_dataset.npy")


    def test_file(self):
        #For each training set resample it and add it to a numpy array
        test_dataset = []
        labels = []
        print(len(self.trainingSet))
        for squatSet in self.testingSet:
            labels.append(squatSet.formName)
            segmented_resampled_squats = np.array([np.pad(segment, ((0, self.longest_squatset_segment() - len(segment)), (0, 0)), mode='constant') for segment in squatSet.each_sginal_segmented])
            segmented_resampled_squats = np.swapaxes(segmented_resampled_squats, 1, 2)
            test_dataset.append(segmented_resampled_squats)
        
        test_dataset = np.concatenate(test_dataset, axis=0)
        test_dataset_labels = np.array([label for label in labels for _ in range(10)])
        #Check if the path exists
        np.save(os.path.join(self.path, "test_dataset_labels"), test_dataset_labels)
        np.save(os.path.join(self.path, "test_dataset"), test_dataset)
        self.testingSetPath = os.path.join(self.path, "test_dataset.npy")
