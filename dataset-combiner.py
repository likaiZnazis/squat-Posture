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
import traceback
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
fileName = "IMUData22.csv"
pathToCSVFie = os.path.join(path, fileName)

def creating_file(pathFile):
    open(pathFile, "w", newline="")

def appending_file(pathFile):
    return open(pathFile, "a", newline="")

def closing_file(file):
    file.close()

def file_exists(filecsvName):
    for file in os.listdir(path):
        if(file == filecsvName):
            print("File " + filecsvName + " exists")
            return True
        else:
            print("File " + filecsvName + " does not exist")
            return False

global currentFile
global fileWriter

def divide_set():
    try:
        mainSet = np.loadtxt(pathToCSVFie, dtype=float, skiprows=1, delimiter=",")
        #placing it into 3 separate files
        setRows = mainSet.shape[0] - (mainSet.shape[0] % 3) #Need to make 3 files out of that single file
        indexes = [math.ceil(setRows / 3) * line for line in range(4)] #Break row points
        indexes = indexes[1:-1] #669, 1388
        print(indexes)
        indexi = 1

        #with ensures that the file is automatically closed when no longer needed
        with open(pathToCSVFie, newline='') as csvfile:
            lineReader = csv.reader(csvfile)
            
            for index, row in enumerate(lineReader):
                #filename and its location, this is for checking if the file exists and creating a path for the file
                filecsvName = str(indexi) + "IMU.csv"
                pathToCurrentFile = os.path.join(path, filecsvName)
                if(index==1):
                    #skip the header
                    #Look if a file exists if it does not exist we create a new file
                    #If it does exist we write to it
                    if(not file_exists(filecsvName)):
                       #create file
                       creating_file(pathToCurrentFile)
                       currentFile = appending_file(pathToCurrentFile)
                       fileWriter = csv.writer(currentFile)
                       fileWriter.writerow(row)
                       indexi += 1
                    else:
                        #write the current row to the file
                        currentFile = appending_file(pathToCurrentFile)
                        fileWriter = csv.writer(currentFile)
                        fileWriter.writerow(row)
                        indexi += 1
                    continue
                for breakIndex in indexes:
                    #for each break number we create a new file
                    if(breakIndex == index):
                        #close the current file
                        closing_file(currentFile)
                        indexi += 1
                        print(filecsvName)
                        #Look if a file exists if it does not exist we create a new file
                        if(not file_exists(filecsvName)):
                            #create file
                            creating_file(pathToCurrentFile)
                            currentFile = appending_file(pathToCurrentFile)
                            fileWriter = csv.writer(currentFile)
                            fileWriter.writerow(row)
                            # print("second hello")
                            pass
                        else:
                            pass
                        
                        #create a new file
                        #add the current row
                        
                # Keep adding rows
                else:
                    # print("Ņo")
                    pass
        
    except OSError as err:
        print("OS error occured " + "probably the path is not correct" + " " + str(err))
        print(traceback.format_exc())
    except Exception as err:
        print("Dont know how to handle this error " + str(err))
        print(traceback.format_exc())
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