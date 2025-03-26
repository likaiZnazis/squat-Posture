import os
"""
Saprast pa daļām kā strādās sistēma.
Izveidojam mapi, kur tiks ielikti mērijumu un markejumu faili. Merijumi notiks ik pa 10 izpildijumiem
    ~/Documents/Squat-classifcation/Measurments config.txt/1/IMUData.csv
    ~/Documents/Squat-classifcation/Measurments config.txt/2/IMUData.csv 
                                                           12
    
    Faili ir izveidoti, lai veiktu apstradi
    *notiek apstrade*

    Lietas kas mainas
    path = "C:/Users/maris/Downloads/IMUData22.csv"

    fs = 50 Hz

    fourier_cutoff = 20 Hz

    segmented_squats = segmented_squats[1:11]

    label_to_number = {
    "bad-squat": 0,
    "good-squat": 1,
    }

    squat_labels = ["bad-squat", "bad-squat", "bad-squat", "bad-squat", "good-squat", "good-squat", "good-squat", "bad-squat", "good-squat", "good-squat"]

"""