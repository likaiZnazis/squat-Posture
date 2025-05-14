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
    # trainingSetPath: str
    # testingSetPath: str
    path: str = os.path.join(os.getcwd(), "dataset")

    def getFrequency(self) -> int:
        # #WE load the dataset
        # data = np.loadtxt(dataset,dtype="float", skiprows=1, delimiter=",")
        # #Grab timestamps
        # timestamps = data[:,0]
        # #Get differences between each point
        # #For example 1743403873.08 - 1743403873.10 = 
        # timeStampDifference = np.diff(timestamps)
        # #Get common value
        # return stats.mode(timeStampDifference)
        pass
    
    def combine_all_sets(self) -> None:
        try:
            # Get path to the set files. They will start with a number
            pathToSetFiles = [ os.path.join(self.path, file) for file in os.listdir(self.path) if (file[0].isdigit())]
            # Get segments from each file
            for pathSetFile in pathToSetFiles:
                newSet = SquatSet(fileName=pathSetFile.split("\\")[-1])
                #load the file
                newSet.formName = pathSetFile.split("\\")[-1].split("-")[1]
                currentFile = np.loadtxt(pathSetFile,dtype="float", skiprows=1, delimiter=",")
                #Get segment indexes
                segment_indexes = preprocessing.get_segment_indexes(currentFile)
                newSet.segmentAmountBefore = len(segment_indexes)
                #Extract each sensor signal from the segment indexes
                newSet.each_sginal_segmented = preprocessing.extract_each_signal(segment_indexes, currentFile)
                newSet.segmentAmountAfter = len(newSet.each_sginal_segmented)
                self.allSets.append(newSet)
            
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
        # train_dataset_labels = np.concatenate(labels, axis=0)
        # print(train_dataset_labels.shape)
        # np.save(os.path.join(self.path, "train_dataset_labels"), train_dataset_labels)
        # np.save(os.path.join(self.path, "train_dataset"), train_dataset)


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
        # test_dataset_labels = np.concatenate(labels, axis=0)
        # print(test_dataset_labels.shape)
        # np.save(os.path.join(self.path, "test_dataset_labels"), test_dataset_labels)
        np.save(os.path.join(self.path, "test_dataset"), test_dataset)

# dataset = Dataset()
# dataset.combine_all_sets()
# dataset.split_sets()
# dataset.test_file()
# dataset.train_file()