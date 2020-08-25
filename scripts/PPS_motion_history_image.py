import os, cv2
import numpy as np
import time
start_time = time.time()

datapath = '../data/'
folder = 'batch10_R1_LEO_SB20_earth_8frames'
animation = 'animation19_R1.0_S6900ms_orbitDiff_65109m.avi'

vidcap = cv2.VideoCapture(os.path.join(datapath, folder, animation))
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

blended_mask = []
for n in range(frames):
    success, img = vidcap.read()

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to colorspace
    lower_bright = np.array([0 * (179 / 360), 0 * (255 / 100), 40 * (255 / 100)])  # HSV maxRange: [179, 255, 255]
    upper_bright = np.array([0 * (179 / 360), 0 * (255 / 100), 100 * (255 / 100)])
    mask = cv2.inRange(hsvImg, lower_bright, upper_bright)

    Bstep = 1 / frames  # brightness change per iteration(frame)
    Bfactor = Bstep * n  # brightness factor

    if n == 0:
        indices = np.nonzero(mask)
        blended_mask = mask
        blended_mask[indices[0], indices[1]] = Bfactor * 255
    else:
        indices = np.where(mask == 255)  # find most recent mask pixel locations
        blended_mask[indices[0], indices[1]] = Bfactor * 255  # set most recent mask brightness

    if Bfactor < 0.1:
        # brighteness factor too low, set to constant in order to encode
        # temporal infomation of the whole motion
        Bfactor = 0.1
        blended_mask[indices[0], indices[1]] = Bfactor * 255  # adjust blended mask

print("--- %s seconds ---" % (time.time() - start_time))
cv2.imwrite('MHI_Earth.png', blended_mask)
