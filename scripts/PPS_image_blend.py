import os, cv2
import time
start_time = time.time()

datapath = '../data/'
folder = 'batch10_R1_LEO_SB20_earth_8frames'
animation = 'animation19_R1.0_S6900ms_orbitDiff_65109m.avi'

vidpath = os.path.join(datapath, folder, animation)

vidcap = cv2.VideoCapture(vidpath)
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

blended_img = []
for n in range(frames):
    success, frame = vidcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to Grayscale

    alpha = (n+1)/frames

    if n == 0:
        blended_img = frame
    else:
        blended_img = cv2.addWeighted(blended_img, alpha, frame, 1, 0) # works well on stars
        #blended_img = cv2.addWeighted(blended_img, alpha, frame, 1 - alpha, 0) # works well on Earth

    #imgS = cv2.resize(blended_img, (960, 540))
    #cv2.imshow('blended_img', imgS)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
cv2.imwrite('blended_stars_test.png', blended_img)


# Test alpha values
#def alpha(n, frames=8):
#     alpha = (n+1)/frames
#     return alpha

#for n in range(8):
#    print('alpha for n={}: {}'.format(n, alpha(n)))




