import os
import shutil

def CopyFiles(Filenames, folder, outpath):
    # Make folders from respective filepaths
    types = ['.png', '.txt']
    CopyPath = os.path.join(outpath, folder)
    if os.path.exists(CopyPath):
        pass
    else:
        os.mkdir(CopyPath)
    for i in range(len(types)):
        for x in range(len(Filenames)):
            shutil.copy(Filenames[x]+types[i], CopyPath)

# Execute YOLOconfig_dataset.py first and use output train/test.txt files to make dataset in Faster R-CNN format.
# 1. Read darknetDir train/test.txt files
# 2. Extract filenames from .txt files
# 3. Navigate to dataset directory and copy train/test filenames in respective folders.

darknetDir = '/vol/teaching/smilon/Repos/darknet/data' # Specify where train/test.txt is saved
datapath = '/vol/teaching/smilon/data'
outpath = '/user/HS122/ml00503/dissertation/data/Faster_R-CNN'

# Read filenames from .txt files
trainF = open(os.path.join(darknetDir, 'train.txt'), 'r')
trainFilenames = trainF.readlines()
trainFilenames = [line[0:-5] for line in trainFilenames] # remove newline character
trainF.close()

testF = open(os.path.join(darknetDir, 'test.txt'), 'r')
testFilenames = testF.readlines()
testFilenames = [line[0:-5] for line in testFilenames] # remove newline character
testF.close()

# Copy Files to folders
#CopyFiles(trainFilenames, 'train', outpath)
CopyFiles(testFilenames, 'test', outpath)



