import cmd
# import os
from test import TestPreprocessig
import unittest
import dataset_combiner

class CLI(cmd.Cmd):
    prompt = "->"

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
        dataset_combiner.main()

        print("Finished combining sets")
        print("New file - final_dataset.npy was created at directory - dataset")

    #Command that will split the dataset into training and testing
    def do_split(self, line):
        pass

    #Command that will train and test the module. Return a word file containing all the statistics
    def do_train(self, line):
        pass

if __name__ == '__main__':
    CLI().cmdloop()