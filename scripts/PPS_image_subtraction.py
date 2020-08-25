import os, cv2
import numpy as np
import time
import glob

start_time = time.time()

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

datapath = '../data'
folder = 'batch17_R0.5_LEO_SB30_8frames'
animation = 'animation5_R0.5_S6900ms_orbitDiff_63934m.avi'
#animation = 'animation16_R1.0_S6900ms_orbitDiff_76388m.avi'

vidpath = os.path.join(datapath, folder, animation)

# Perform inference
#backsub = cv2.createBackgroundSubtractorKNN(detectShadows=False) # init KNN backsub
backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=False) # init MOG2 backsub
blended_img = backsubForwardPass(backsub, vidpath)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imwrite('IS_MOG2_Earth.png', blended_img)
