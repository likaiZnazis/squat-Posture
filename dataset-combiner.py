#What I need
"""""
Let's assume that we have a folder where are 12 csv files.
We need to rename each of the files to 1-Set.csv, 2-Set.csv etc
Need to log each of the files preprocessing where are segments made.
These segments are placed insdie a single file.
"""""
import os
import numpy as np
import csv
import math
#Steps
'''
FIRST: need to understand where to place the file. I don't want to change the path each time.
For example if I pull this git to a different PC then it would create a folder and place these csv files.
-Place it inside the project direcotry. This happens on github ;/

SECOND: need to go over each of the files and get segments of those files.
Log information about the segments and append them to the main csv file.
'''

#Command shortcut
'''
#getcwd() - gets current working directory
#mkdir(path) - creates a directory
# os.mkdir(currentDirectory + "\dataset")
#path.join() - join a bunch of string together to form a path
#rmdir - removes a directory
# os.rmdir(path)
'''

activeDirectoryName = "dataset"
currentDirectory = os.getcwd()
path = os.path.join(currentDirectory, activeDirectoryName)
fileName = ''.join(os.listdir(path))
pathToCSVFie = os.path.join(path, fileName)
# print(pathToCSVFie)


# indexi = 1
# filecsvName = str(indexi) + "IMU.csv"
# currentCSVfile = os.path.join(path, filecsvName)
# with open(path, newline="") as newcsv:
#     lineWriter = csv.writer(path)
# def creating_file(pathFile):
#     with open(pathFile, "w", newline=""):
#         pass
# def writing_file(pathFile):
#     return open(pathFile, "w", newline="")

# def closing_file(file):
#     file.close()

# file = writing_file(currentCSVfile)


def divide_set():
    try:
        mainSet = np.loadtxt(pathToCSVFie, dtype=float, skiprows=1, delimiter=",")
        #placeing it into 3 separate files
        setRows = mainSet.shape[0] - (mainSet.shape[0] % 3) #Need to make 3 files out of that single file
        indexes = [math.ceil(setRows / 3) * line for line in range(4)] #Break row points
        indexes = indexes[:-1] #Dont need the last one
        indexi = 1
        # filecsvName = str(indexi) + "IMU.csv"
        currentCSVfile = os.path.join(path, filecsvName)
        
        #Look if a file exists if it does not exist we create a new file
        print(os.listdir(path))
        # creating_file(currentCSVfile)

        #with ensures that the file is automatically closed when no longer needed
        with open(pathToCSVFie, newline='') as csvfile:
            lineReader = csv.reader(csvfile)
            
            for index, row in enumerate(lineReader):
                filecsvName = str(indexi) + "IMU.csv"
                if(index<1):
                    #skip the header
                    continue
                for breakIndex in indexes:
                    if(breakIndex == index):
                        #for each break number we create a new file
                        indexi += 1
                        # print(indexi)
                        # fileName = str(indexi) + "IMU.csv"
                        print(filecsvName)
                        #close the current file
                        #create a new file
                        #add the current row
                        
                # Keep adding rows
                else:
                    # print("Ņo")
                    pass
                # else:
                #     print(lineReader.line_num)
                #     print(index)
                #     print(", ".join(row))
                    
                    # Returns a array - print(row)

        # for line in csvFile - setRows:
        #     print()
        #Loop over the whole set c
        # for 
        
    except OSError as err:
        print("OS error occured " + "probably the path is not correct" + " " + str(err))
    except Exception as err:
        print("Dont know how to handle this error " + str(err))
        # raise - reraising the error is only useful if there are higher-up try,except
    except ValueError:
        print("")
    pass

divide_set()

def creat_single_dataset():
    #SECOND step
    #First need to get the file into the dataset folder
    #Divide the file into 3 parts
        #
    pass



def creating_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)