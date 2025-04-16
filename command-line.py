import cmd
# import os
from test import TestPreprocessig
import unittest
import dataset_combiner
import word_report

class CLI(cmd.Cmd):
    prompt = "->"
    
    #This information is for word report
    wordReport = {
        "total_records": 0,
        "sensor_freq": 0,
        "segment_length": 0,
        "form_count": {
            "forma_1": 0,
            "forma_2": 0,
        },
        "squat_processing":{
            "IMU-1":
            {
                "before": 0,
                "after": 0
            }
        }
    }
    intro = "Squat classification"

    #Command that will test all of the sets
    def do_tests(self, line):
        "Test how many reps are extracted from each squat set"
        print("Testing sets")
        #TestSuite to run tests in isolation
        suite = unittest.TestSuite()
        #Create a runner instace
        runner = unittest.TextTestRunner()
        #Can add a bunch of tests that we want to check
        suite.addTest(TestPreprocessig("test_shapeSET"))
        runner.run(suite)
        print("Finish testing")

    #Command will combine all the set files inside a single numpy file, also print how many reps are inside
    def do_combine(self, line):
        "Combine all the sets together into a single .npy file. Should run command 'tests' before"
        print("Starting to combine sets")
        datasetInfo = dataset_combiner.main()
        print(datasetInfo)#(segmenti, iezimes, merijumu paraugi)
        print("Finished combining sets")
        print("New file - final_dataset.npy was created at directory - dataset")

    #Command that will split the dataset into training and testing
    def do_split(self, line):
        pass

    #Command that will train and test the module. Return a word file containing all the statistics
    def do_train(self, line):
        pass

    #Need to check if all the variables are set
    def do_report(self, line):
        "Create a word report when the model has been trained"
        # word_report.create_report()

if __name__ == '__main__':
    CLI().cmdloop()