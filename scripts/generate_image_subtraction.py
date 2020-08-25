# Author @ Milton Logothetis (milton.logothetis@gmail.com)
# Date: 05/08/2020
#
# Description:
# Performs MOG2 background subtraction on all '.avi' content on a specified folder, while also copying the annotation
# files to the specified output folder. Choose to either train background subtractor and then perform inference
# (uncomment line 144) or just perform a single forward pass. Proceeds to augment each output image and its respective
# annotation file with the transformations specified in the transformation dict (horizontal, vertical and
# vertical/horizontal flips).
#
# Inputs:
# 1. Path to directory containing '.avi' folder.
# 2. Folder to '.avi' directory.
# 3. Output path.

import os, cv2
import numpy as np
from scipy import ndarray
import time, shutil
import glob


def horizontal_flip(image_array: ndarray, annotationOld, annotationNew):
    shutil.copy(annotationOld, annotationNew)
    r = open(annotationOld, 'r')
    label, x, y, width, height = r.read().split(' ')
    x = str(1 - float(x))  # modification for horizontal flip
    r.close()
    w = open(annotationNew, 'w')
    w.write(label + ' ' + x + ' ' + y + ' ' + width + ' ' + height + '\n')
    w.close()
    return np.flip(image_array, 1)

def vertical_flip(image_array: ndarray, annotationOld, annotationNew):
    shutil.copy(annotationOld, annotationNew)
    r = open(annotationOld, 'r')
    label, x, y, width, height = r.read().split(' ')
    y = str(1 - float(y))  # modification for vertical flip
    r.close()
    w = open(annotationNew, 'w')
    w.write(label + ' ' + x + ' ' + y + ' ' + width + ' ' + height + '\n')
    w.close()
    return np.flip(image_array, 0)

def VH_flip(image_array: ndarray, annotationOld, annotationNew):
    shutil.copy(annotationOld, annotationNew)
    r = open(annotationOld, 'r')
    label, x, y, width, height = r.read().split(' ')
    x = str(1 - float(x))  # modification for horizontal flip
    y = str(1 - float(y))  # modification for vertical flip
    r.close()
    w = open(annotationNew, 'w')
    w.write(label + ' ' + x + ' ' + y + ' ' + width + ' ' + height + '\n')
    w.close()

    h_flip = np.flip(image_array, 1)
    return np.flip(h_flip, 0)

def backsubTrain(vidpath):
    # Choose subtractor type (MOG2 faster & requires less training frames)
    # backsub = cv2.createBackgroundSubtractorKNN()
    backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Decompose video to images
    vidcap = cv2.VideoCapture(vidpath)
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Train background subtractor
    for n in range(frames):
        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to Grayscale
        subtracted_img = backsub.apply(frame)

        # Illustrate Subtractor Learning Process
        #imgS = cv2.resize(subtracted_img, (960, 540))
        #cv2.imshow('Learning epoch: {}'.format(n), imgS)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return backsub

def backsubForwardPass(backsub, vidpath):
    # Find if backsub passed in function
    if 'backsub' in locals():
        pass
    else:
        # Initialize backsub
        backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Perform inference with trained subtractor or train if (backsub in locals == False)
    vidcap = cv2.VideoCapture(vidpath) # re-initialize vidcap
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for n in range(frames):
        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to Grayscale
        subtracted_img = backsub.apply(frame)

        # Illustrate Subtractor Inference Process (useful to find minimum input frames for optimal background removal)
        #imgS = cv2.resize(subtracted_img, (960, 540))
        #cv2.imshow('Inference frame: {}'.format(n), imgS)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        Bstep = 1/frames
        Bfactor = n*Bstep

        if Bfactor < 0.1:
            # Set lower limit to brightness reduction factor
            Bfactor = 0.1

        # Assuming debris will be the brightest artifact
        if n == 0:
            indices = np.nonzero(subtracted_img)
            blended_img = subtracted_img
            blended_img[indices[0], indices[1]] = Bfactor * 255
        else:
            indices = np.where(subtracted_img == 255)  # find most recent mask pixel locations
            blended_img[indices[0], indices[1]] = Bfactor * 255  # set most recent mask brightness
    return blended_img


#****************  M A I N  S C R I P T  ****************#

datapath = '../data'
folder = 'batch25_R1_LEO_SB50_8frames'
outpath = '/data/' + folder

if os.path.exists(outpath):
    pass
else:
    os.mkdir(outpath)

files = os.listdir(os.path.join(datapath, folder))
transformations = {
    'H_Flip': horizontal_flip,
    'V_Flip': vertical_flip,
    'VH_Flip': VH_flip,
}

for i in range(0, len(files)):
    start_time = time.time()

    # Apply background subtraction to animations
    if files[i].endswith('.avi'):
        vidpath = os.path.join(datapath, folder, files[i])
        # Train background subtractor
        #backsub = backsubTrain(vidpath)
        # Perform inference
        backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False) # init backsub
        blended_img = backsubForwardPass(backsub, vidpath)
        cv2.imwrite(os.path.join(outpath, files[i][0:-4]+'.png'), blended_img)
        print("--- %s seconds ---" % (time.time() - start_time))

    # Copy annotations in outpath
    elif files[i].endswith('.txt'):
        shutil.copy(os.path.join(datapath, folder, files[i]), outpath)

# Augment background subtracted output with transformations[dict]
imagefiles = []
annotations = []
for image in glob.glob(outpath+'/*.png'):
    imagefiles.append(image)
for annotation in glob.glob(outpath+'/*.txt'):
    annotations.append(annotation)

for i in range(len(imagefiles)):
    filename = imagefiles[i][0:-4]
    img = cv2.imread(os.path.join(outpath, imagefiles[i]))
    for n in range(len(transformations)):
        key = list(transformations)[n]
        annotationNew = filename + '_' + key + '.txt'
        img_transformed = transformations[key](img, os.path.join(outpath,annotations[i]),os.path.join(outpath,annotationNew))
        cv2.imwrite(filename + '_' + key + '.png', img_transformed)
