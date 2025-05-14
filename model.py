import numpy as np
import os
import time
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from sklearn import metrics

class ClassifierModel():

    training_set_labels_path = os.path.join(os.getcwd(), "dataset","train_dataset_labels.npy")#need to change
    testing_set_lables_path = os.path.join(os.getcwd(), "dataset","train_dataset_labels.npy")#need to change
    training_set_path = os.path.join(os.getcwd(), "dataset","train_dataset.npy")
    testing_set_path = os.path.join(os.getcwd(), "dataset","test_dataset.npy")

    #katrai modela instancei tiek izveidoti atseviski mainigie
    def __init__(self):
        self.accuracy = None
        self.specificity = None
        self.sensitivity = None
        self.classifier = MultiRocketHydraClassifier(random_state=0)
        self.confusion_matrix = None
        self.training_time = None

    def train_model(self):
        training_set = np.load(self.training_set_path)
        training_set_labels = np.load(self.training_set_labels_path)
        start_time = time.time()
        self.classifier.fit(training_set, training_set_labels)
        end_time = time.time()
        self.training_time = end_time - start_time

#think of ways to evaluate the model
    def evaluate_model(self):
        self.classifier.fit
        pass