This repository is designed to preprocess and classify IMU data collected from the IMUReader repo - https://github.com/likaiZnazis/IMUReader.

The IMUReader app records:

    Acceleration and Gyroscope data across all axes

    Orientation angles: Roll, Pitch, and Yaw

How to Prepare Your Dataset?

Once you’ve collected measurements and exported them as .csv files from the IMUReader app:

    Label your files properly using the following naming format:

{order number}-{label}-Set.csv

Example:

    1-Full-Set.csv

        Important: This naming format is used to extract the label automatically. It must be followed precisely.

    Place all .csv files inside the dataset/ folder in this repository.

    Ensure your dataset is balanced – each class (label) should have the same number of repetitions (segments). This will be made sure later.

Installation

Install the required dependencies:

python -m pip install -r requirements.txt

    Note: To view graphs, you may need to install and configure matplotlib separately depending on your environment.

Running the Command-Line Interface

Start the preprocessing pipeline by running:

python -m command-line.py

This will launch a Command-Line Interface (CLI).

To see all available commands, type:

help

Command Sequence

Run the following commands in the given order:

    1tests
    Verifies all .csv files and checks that each contains exactly 10 segments (repetitions). Each segment represents an exercise repetition.

    2file
    Converts .csv files into .npy files. Stores all segments and their labels. Splits the data into training and testing datasets.

    3train
    Trains a MULTIROCKET-Hydra classifier using the training dataset.

    4predict
    Evaluates the trained model using the testing dataset.

    5report
    Generates a detailed report on model performance and dataset statistics.
