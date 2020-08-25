import os, cv2
import time
start_time = time.time()

datapath = '../data/'
folder = 'batch10_R1_LEO_SB20_earth_8frames'
animation = 'animation16_R1.0_S6900ms_orbitDiff_76388m.avi'
vidpath = os.path.join(datapath, folder, animation)

vidcap = cv2.VideoCapture(vidpath)
frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

def method1(vidcap, frames):
    # https://stackoverflow.com/questions/31377832/simulate-long-exposure-from-video-frames-opencv
    _exposed = []
    for n in range(frames):
        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to Grayscale

        alpha = 1/frames

        if n == 0:
            _exposed = frame
            _exposed = cv2.addWeighted(_exposed, 0, frame, alpha, 0)
        else:
            _exposed = cv2.addWeighted(_exposed, 1, frame, alpha, 0)

        #imgS = cv2.resize(_exposed, (960, 540))
        #cv2.imshow('blended_img', imgS)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    print("--- %s seconds ---" % (time.time() - start_time))
    cv2.imwrite('long_exposure_method1_stars.png', _exposed)


def method2(vidcap, frames):
    # https://www.pyimagesearch.com/2017/08/14/long-exposure-with-opencv-and-python/
    for n in range(frames):
        success, frame = vidcap.read()
        (B, G, R) = cv2.split(frame.astype("float"))

        if n == 0:
            rAvg = R
            gAvg = G
            bAvg = B

        else:
            rAvg = ((n * rAvg) + (1 * R)) / (n + 1.0)
            gAvg = ((n * gAvg) + (1 * G)) / (n + 1.0)
            bAvg = ((n * bAvg) + (1 * B)) / (n + 1.0)

    avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")
    print("--- %s seconds ---" % (time.time() - start_time))
    cv2.imwrite('long_exposure_method2_stars.png', avg)


# Execute
#method1(vidcap, frames)
method2(vidcap, frames)