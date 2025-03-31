import unittest
import os
import numpy as np
import dataset_combiner
import preprocessing

class TestPreprocessig(unittest.TestCase):
    _pathToDict = os.path.join(os.getcwd(),"dataset")

    #Testing each file shape
    def test_shapeSET(self):
        pathToSetFiles = [ os.path.join(self._pathToDict, file) for file in os.listdir(self._pathToDict) if (file[0].isdigit())]
        for setFile in pathToSetFiles:
            currentFile = np.loadtxt(setFile,dtype="float", skiprows=1, delimiter=",")
            segment_indexes = preprocessing.get_segment_indexes(currentFile)
            sensor_signals = preprocessing.extract_each_signal(segment_indexes, currentFile)
            segmented_resampled_set =  preprocessing.resample_segments(sensor_signals)
            file_dimenstions = segmented_resampled_set.shape
            with self.subTest(line = file_dimenstions):
                self.assertEqual(file_dimenstions, (10, 13, file_dimenstions[2]))

    #Testing the final numpy shape
    def test_shapeFINAL(self):
        if(not dataset_combiner.file_exists("final_dataset")):
            dataset_combiner.main()
        final_dataset = np.load(os.path.join(self._pathToDict, "final_dataset.npy"))
        dataset_dimensions = final_dataset.shape
        self.assertEqual(dataset_dimensions, (20, 13, dataset_dimensions[2]))

if(__name__ == "__main__"):
    unittest.main()