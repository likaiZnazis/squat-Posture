import cmd
# import os
from test import TestPreprocessig
import unittest

class CLI(cmd.Cmd):
    prompt = "->"

    intro = "Squat classification"
        
    #Command that will test all of the sets
    def do_tests(self, line):
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
    #Command that will split the dataset into training and testing
    #Command that will train and test the module. Return a word file containing all the statistics
    pass

if __name__ == '__main__':
    CLI().cmdloop()