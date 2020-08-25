import os
import random

def shuffleLines(path):
    lines = open(path).readlines()
    random.shuffle(lines)
    open(path, "w").writelines(lines)

darknetDir = ''
datapath = '/data/FinalSet'
data = os.listdir(datapath)

# Arrange into images and labels lists
images = []
labels = []
for i in range(len(data)):
    if data[i].endswith('.txt'):
        labels.append(data[i])
    else:
        images.append(data[i])
imageformat = images[0][-4:]

# Extract filenames (using list compression)
filenames = [x[:-4] for x in images]

# Split into train/test datasets
trainSplit = 85
testSplit = 100 - trainSplit

trainData = random.sample(list(enumerate(filenames)), round(trainSplit*len(filenames)/100)) # returns [index, filename]
totalIndices = list(range(len(filenames))) # total indices
trainIndices = [x[0] for x in trainData]
testIndices = list(set(totalIndices).difference(trainIndices))

# Write train.txt file
with open(os.path.join(darknetDir, 'train.txt'), 'w') as f:
    for i in range(len(trainIndices)):
        path = datapath+'/'+filenames[trainIndices[i]]+imageformat+'\n'
        f.write(path)

shuffleLines(os.path.join(darknetDir, 'train.txt'))
f.close()

# Write test.txt file
with open(os.path.join(darknetDir, 'test.txt'), 'w') as f:
    for i in range(len(testIndices)):
        path = datapath+'/'+filenames[testIndices[i]]+imageformat+'\n'
        f.write(path)

shuffleLines(os.path.join(darknetDir, 'test.txt'))
f.close()

