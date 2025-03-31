import unittest
import os
import numpy as np
import dataset_combiner
import preprocessing

class TestPreprocessig(unittest.TestCase):
    _pathToDict = os.path.join(os.getcwd(),"dataset")

    #Testing each file shape
    def test_shapeSET(self):
        for setFile in self._pathToDict:
            currentFile = np.loadtxt(setFile,dtype="float", skiprows=1, delimiter=",")
            file_dimenstions = (preprocessing.extract_each_signal(preprocessing.get_segment_indexes(currentFile))).shape
            with self.subTest(line = file_dimenstions):
                self.assertEqual(file_dimenstions, (10, 13, file_dimenstions.shape[2]))

    #Testing the final numpy shape
    def test_shapeFINAL(self):
        dataset_combiner.combine_sets_numpy()
        final_dataset = np.loadtxt(os.path.join(self._pathToDict, "final_dataset"), dtype="float", delimiter=",")
        dataset_dimensions = final_dataset.shape
        self.assertEqual(dataset_dimensions, (120, 13, dataset_dimensions.shape[2]))