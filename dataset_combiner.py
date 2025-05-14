
import os
import numpy as np
import traceback
import preprocessing
# from scipy import stats

activeDirectoryName = "dataset"
currentDirectory = os.getcwd()
path = os.path.join(currentDirectory, activeDirectoryName)

def combine_sets():
    try:
        # Get path to the set files. They will start with a number
        pathToSetFiles = [ os.path.join(path, file) for file in os.listdir(path) if (file[0].isdigit())]
        # Get segments from each file
        all_signals = []
        for pathSetFile in pathToSetFiles:
            #load the file
            currentFile = np.loadtxt(pathSetFile,dtype="float", skiprows=1, delimiter=",")
            #Get segment indexes
            segment_indexes = preprocessing.get_segment_indexes(currentFile)
            #Extract each sensor signal from the segment indexes
            sensor_signals = preprocessing.extract_each_signal(segment_indexes, currentFile) 
            #sensor_signals is a 2D array
            for row in sensor_signals:
                all_signals.append(row)
        return all_signals

    except OSError as err:
        print("OS error occured " + "probably the path is not correct" + " " + str(err))
        print(traceback.format_exc())
    except Exception as err:
        print("Dont know how to handle this error " + str(err))
        print(traceback.format_exc())
        # raise - reraising the error is only useful if there are higher-up try,except
    except ValueError:
        print("There was a value error " + str(err))
        print(traceback.format_exc())

def delete_files(fileName):
    for file in os.listdir(path):
        if(file != fileName):
            os.remove(os.path.join(path, file))
        else:
            continue

def file_exists(fileName):
    for file in os.listdir(path):
        if(file == fileName):
            return True
        else:
            return False

# def getFrequency(dataset):
#     #WE load the dataset
#     data = np.loadtxt(dataset,dtype="float", skiprows=1, delimiter=",")
#     #Grab timestamps
#     timestamps = data[:,0]
#     #Get differences between each point
#     #For example 1743403873.08 - 1743403873.10 = 
#     timeStampDifference = np.diff(timestamps)
#     #Get common value
#     return stats.mode(timeStampDifference)

# preprocessing.show_graph(os.path.join(path, "1-Set.csv"))
def main():
    extracted_segments = combine_sets()
    dataset = preprocessing.resample_segments(extracted_segments)
    # getFrequency(dataset)
    print("There are {} segments total".format(dataset.shape[0]))
    np.save(os.path.join(path, "final_dataset"), dataset)
    return dataset.shape
#working with csv files
"""
import csv
import math
def closing_file(file):
    file.close()

def creating_file(pathFile):
    file = open(pathFile, "w", newline="")
    #close the file besauce we are just creating it
    closing_file(file)

def appending_file(pathFile):
    return open(pathFile, "a", newline="")

def combine_sets_file():
    try:
        # Get path to the set files. They will start with a number
        pathToSetFiles = [ os.path.join(path, file) for file in os.listdir(path) if (file[0].isdigit())]
        #path to the combined set file
        combinedFilePath = os.path.join(path, "combinedIMU.csv")
        #if combined file does not exist create it
        if(not file_exists("combinedIMU.csv")):
            creating_file(combinedFilePath)
        
        #open the created file
        with open(combinedFilePath, 'a', newline='') as csvFile:
            lineWriter = csv.writer(csvFile)
            #for each of the set files
            for pathFile in pathToSetFiles:
                #open up those files to read them
                print("Reading from path: " + pathFile)
                with open(pathFile, newline="") as setFile:
                    lineReader = csv.reader(setFile)
                    next(lineReader)
                    for index, row in enumerate(lineReader, start=1):
                        lineWriter.writerow(row)
    except OSError as err:
        print("OS error occured " + "probably the path is not correct" + " " + str(err))
        print(traceback.format_exc())
    except Exception as err:
        print("Dont know how to handle this error " + str(err))
        print(traceback.format_exc())
        # raise - reraising the error is only useful if there are higher-up try,except
    except ValueError:
        print("There was a value error " + str(err))
        print(traceback.format_exc())

def creating_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
#Need to refactor code that if the file exists we return the file 
def file_needs(pathToCurrentFile, header):
    currFile = appending_file(pathToCurrentFile)
    fileWrite = csv.writer(currFile)
    fileWrite.writerow(header)
    return currFile, fileWrite

global currentFile
global fileWriter
def divide_set():
    try:
        mainSet = np.loadtxt(pathToCSVFie, dtype=float, skiprows=1, delimiter=",")
        #placing it into 3 separate files
        setRows = mainSet.shape[0] - (mainSet.shape[0] % 3) #Need to make 3 files out of that single file
        indexes = [math.ceil(setRows / 3) * line for line in range(4)] #Break row points
        indexes = indexes[1:-1] #669, 1388
        indexi = 1

        #with ensures that the file is automatically closed when no longer needed
        with open(pathToCSVFie, newline='') as csvfile:
            lineReader = csv.reader(csvfile)
            header = next(lineReader)
            for index, row in enumerate(lineReader, start=1):
                #filename and its location, this is for checking if the file exists and creating a path for the file
                filecsvName = str(indexi) + "IMU.csv"
                pathToCurrentFile = os.path.join(path, filecsvName)
                if(index == setRows):
                    #Close the file for the last row, maybe can do it else where
                    closing_file(currentFile)
                    break
                elif(index==1):
                    #Look if a file exists if it does not exist we create a new file
                    #If it does exist we write to it
                    if(not file_exists(filecsvName)):
                       #create file
                       creating_file(pathToCurrentFile)
                       currentFile, fileWriter = file_needs(pathToCurrentFile, header)
                       fileWriter.writerow(row)
                       indexi += 1
                    else:
                        #write the current row to the file
                        currentFile, fileWriter = file_needs(pathToCurrentFile, header)
                        fileWriter.writerow(row)
                        indexi += 1
                    continue
                for breakIndex in indexes:
                    #for each break number we create a new file
                    if(breakIndex == index):
                        #close the current file
                        closing_file(currentFile)
                        indexi += 1
                        #Look if a file exists if it does not exist we create a new file
                        if(not file_exists(filecsvName)):
                            #create file
                            creating_file(pathToCurrentFile)
                            currentFile, fileWriter = file_needs(pathToCurrentFile, header)
                            fileWriter.writerow(row)
                        else:
                            #write to the current file
                            currentFile, fileWriter = file_needs(pathToCurrentFile, header)
                            fileWriter.writerow(row)

                # If the its not the last or first index we add rows to the current file
                else:
                    print(index)
                    fileWriter.writerow(row)
        
    except OSError as err:
        print("OS error occured " + "probably the path is not correct" + " " + str(err))
        print(traceback.format_exc())
    except Exception as err:
        print("Dont know how to handle this error " + str(err))
        print(traceback.format_exc())
        # raise - reraising the error is only useful if there are higher-up try,except
    except ValueError:
        print("There was a value error " + str(err))
        print(traceback.format_exc())
"""