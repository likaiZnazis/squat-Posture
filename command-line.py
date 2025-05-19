import cmd
from test import TestPreprocessig
import unittest
from squat_dataset import Dataset
from model import ClassifierModel
import word_report

class CLI(cmd.Cmd):
    prompt = "->"

    wordReport = {
        #Some data distrubution graphs for data distribution
        "total_records": 0,
        "sensor_freq": 0,
        "longest_segment_length": 0,
        "form_count": {
            # "forma_1": 0,
            # "forma_2": 0,
        },
        #Model metrics
        "accuracy": 0,
        "specificity": 0,
        "sensitivity": 0,
        "confusion_matrix_path": None
        #confusion metrcis, check out sklearn
    }

    intro = "Squat classification"
    
    def __init__(self, completekey = "tab", stdin = None, stdout = None):
        super().__init__(completekey, stdin, stdout)
        self.dataset = Dataset()
        self.model = ClassifierModel()

    #Command that will test all of the squat set csv files
    def do_1tests(self, line):
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
    def do_2file(self, line):
        "Splits all sets into training and testing data and gives files"
        print("Combining all the sets")
        self.wordReport["total_records"] = self.dataset.combine_all_sets()
        self.wordReport["form_count"] = self.dataset.split_sets()
        self.dataset.train_file()
        self.dataset.test_file()
        self.wordFileInfo_Write()
        print("Creating files")
        

    def wordFileInfo_Write(self):
        self.wordReport["longest_segment_length"] = self.dataset.longest_squatset_segment()
        self.wordReport["sensor_freq"] = self.dataset.getFrequency()

    #Command that will train and test the module. Return a word file containing all the statistics
    def do_3train(self, line):
        "Train a MultiRocketHydraClassifier based on data provided"
        self.model.train_model()

    #Command that will test the model labels
    def do_4predit(self, line):
        "Tests the testing data set part"
        self.model.evaluate_model()
        self.wordReport["accuracy"] = self.model.accuracy
        self.wordReport["specificity"] = self.model.specificity
        self.wordReport["sensitivity"] = self.model.sensitivity
        self.wordReport["confusion_matrix_path"] = self.model.confusion_matrixPath

    #Need to check if all the variables are set
    def do_5report(self, line):
        "Create a word report when the model has been trained"
        for key in ["total_records", "sensor_freq", "longest_segment_length", "form_count"]:
            if self.wordReport[key] == 0:
                print("You need to run all the commands with numbers in front.")
                return

        # Ensure model has been trained and evaluated
        if self.model.accuracy is None:
            print("You must train and evaluate the model first (commands 3 and 4).")
            return

        # Create the report
        word_report.create_report(
            total_records=self.wordReport["total_records"],
            sensor_freq=self.wordReport["sensor_freq"],
            segment_length=self.wordReport["longest_segment_length"],
            form_counts=self.wordReport["form_count"],
            accuracy=self.wordReport["accuracy"],
            specificity=self.wordReport["specificity"],
            sensitivity=self.wordReport["sensitivity"],
            confusion_matrixPath=self.wordReport["confusion_matrix_path"]
        )
        print("Word report generated successfully.")
                

if __name__ == '__main__':
    CLI().cmdloop()