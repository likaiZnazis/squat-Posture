#What I need
"""""
Let's assume that we have a folder where are 12 csv files.
We need to rename each of the files to 1-Set.csv, 2-Set.csv etc
Need to log each of the files preprocessing where are segments made.
These segments are placed insdie a single file.
"""""
import os
#Steps
'''
FIRST: need to understand where to place the file. I don't want to change the path each time.
For example if I pull this git to a different PC then it would create a folder and place these csv files.
-Place it inside the project direcotry
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
print(path)

try:
    os.mkdir(path)
except OSError as error:
    print(error)
