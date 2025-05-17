import cmd
from test import TestPreprocessig
import unittest
from squat_dataset import Dataset
from model import ClassifierModel

# import word_report

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
    
    def __init__(self, completekey = "tab", stdin = None, stdout = None):
        super().__init__(completekey, stdin, stdout)
        self.dataset = Dataset()
        self.model = ClassifierModel()

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

    #Command that will split the dataset into training and testing
    def do_file(self, line):
        "Splits all sets into training and testing data and gives files"
        print("Combining all the sets")
        self.dataset.combine_all_sets()
        print("Spliting sets")
        self.dataset.split_sets()
        print("Creating files")
        self.dataset.train_file()
        self.dataset.test_file()

    #Command that will train and test the module. Return a word file containing all the statistics
    def do_train(self, line):
        "Train a MultiRocketHydraClassifier based on data provided"
        self.model.train_model()

    #Command that will test the model labels
    def do_predit(self, line):
        "Tests the testing data set part"
        self.model.evaluate_model()

    #Need to check if all the variables are set
    def do_report(self, line):
        "Create a word report when the model has been trained"
        # word_report.create_report()

if __name__ == '__main__':
    CLI().cmdloop()