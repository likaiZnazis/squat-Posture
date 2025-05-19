import numpy as np
import os
import time
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ClassifierModel:

    def __init__(self):
        base_path = os.path.join(os.getcwd(), "dataset")
        self.training_set_path = os.path.join(base_path, "train_dataset.npy")
        self.training_set_labels_path = os.path.join(base_path, "train_dataset_labels.npy")
        self.testing_set_path = os.path.join(base_path, "test_dataset.npy")
        self.testing_set_labels_path = os.path.join(base_path, "test_dataset_labels.npy")

        self.classifier = MultiRocketHydraClassifier(random_state=0)
        self.accuracy = None
        self.specificity = None
        self.sensitivity = None
        self.per_class_specificity = {}
        self.per_class_sensitivity = {}
        self.confusion_matrixPath = ""
        self.training_time = None

        # Define class names once
        self.class_names = ["Full", "Parallel", "Wide", "Half", "KneeIn", "KneeOut"]

    def train_model(self):
        X_train = np.load(self.training_set_path)
        y_train = np.load(self.training_set_labels_path)
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        # print(f"Training completed in {self.training_time:.2f} seconds.")

    #Evaluates model based on made datasets and their labels
    def evaluate_model(self):
        X_test = np.load(self.testing_set_path)
        y_test = np.load(self.testing_set_labels_path)
        y_pred = self.classifier.predict(X_test)

        # Accuracy
        self.accuracy = self.classifier.score(X_test, y_test)
        # print(f"Accuracy: {self.accuracy:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.confusion_matrix = cm
        num_classes = len(self.classifier.classes_)

        # Specificity & Sensitivity
        specificities = []
        sensitivities = []

        for i in range(num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)

            spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            specificities.append(spec)
            sensitivities.append(sens)

            self.per_class_specificity[self.class_names[i]] = spec
            self.per_class_sensitivity[self.class_names[i]] = sens

        self.specificity = np.mean(specificities)
        self.sensitivity = np.mean(sensitivities)

        # print("\nPer-Class Sensitivity:")
        # for cls in self.class_names:
        #     print(f"  {cls}: {self.per_class_sensitivity[cls]:.4f}")
        
        # print("\nPer-Class Specificity:")
        # for cls in self.class_names:
        #     print(f"  {cls}: {self.per_class_specificity[cls]:.4f}")

        # print(f"\nAverage Sensitivity: {self.sensitivity:.4f}")
        # print(f"Average Specificity: {self.specificity:.4f}")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Konfuzijas matrica")
        plt.tight_layout()
        plt.xlabel("Minētais marķējums")
        plt.ylabel("Patiesais marķējums")
        self.confusion_matrixPath = os.path.join(os.getcwd(), "dataset", "confusion_matrix.png")
        plt.savefig(self.confusion_matrixPath, bbox_inches='tight')
        plt.close()
