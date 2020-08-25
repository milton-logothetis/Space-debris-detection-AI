# Combine video sequences into a single one and superimpose bounding boxes in evaluation frames.

# 1. Split all videos in respective frames and concatenate with the right order
# 2. After first 8 frames (where prediction occurs) superimpose bounding box that moves as camera moves (use transforma-
# rion matrix at camera plane).
# I.e all frames will contain bounding box apart from first 8, and all bounding box locations will be updated in each frame.
# But whenever new prediction occurs, bounding box is updated ONLY with predicted location.
# 3. Join frames into a single video file.

import os, cv2

def extract_annotations(annotationPath, annotationList, frame):
    with open(os.path.join(annotationPath, annotationList)) as f:
        annotations = f.read()
    annotations = list(map(float, annotations.split(' ')))

    image_size = frame.shape[0:2]
    x_min = round((annotations[1] - annotations[3] / 2) * image_size[0])
    y_min = round((1 - (annotations[2] - annotations[4] / 2)) * image_size[1])
    x_max = round(x_min + (annotations[3] * image_size[0]))
    y_max = round(y_min - (annotations[4] * image_size[1]))
    return (x_min, y_min, x_max, y_max)

fps = 60
OSF = 40 # specify OffSet Factor for a more visible bbox [pixels]
videoFolder = 'C:/Users/Smilon/Desktop/Academic/Uni/Surrey/AI for space debris detection & orbit modelling/data/debris_tracking_animations'
files = os.listdir(videoFolder)

# Camera/debris distance vector at different epochs
DE = []

# Separate true label files from animations (true labels = actual object location)
animations = []
trueAnnot = []
for i in range(len(files)):
    if files[i].endswith('.avi'):
        animations.append(files[i])
    elif files[i].endswith('.txt'):
        trueAnnot.append(files[i])

# Extract frames
total_frames = []
for i in range(len(animations)):
    vidcap = cv2.VideoCapture(os.path.join(videoFolder, animations[i]))
    Nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for n in range(Nframes):
        success, frame = vidcap.read()

        if n == Nframes-1:
            # Read true annotations
            x_min, y_min, x_max, y_max = extract_annotations(videoFolder, trueAnnot[i], frame)
            # Draw bbox
            frame = cv2.rectangle(frame,(x_min-OSF,y_min-OSF),(x_max+OSF,y_max+OSF),(0,255,0),3) # add rectangle to image
        total_frames.append(frame)

height, width, channels = frame.shape
size = (width, height)
out = cv2.VideoWriter(os.path.join(videoFolder, 'combined.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(total_frames)):
    out.write(total_frames[i])
out.release()

# 1. Read DE vector from a .txt file output from Simulation.blend (DE recorded at last frame of every eval period)
# 2. Read satellite position from .txt file output from Blender
# 3. Create a function that uses DE, location of satellite to calculate position,size & velocity of debris



