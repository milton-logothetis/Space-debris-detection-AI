import os, cv2, json
import numpy as np

# Combine video sequences into a single one and superimpose bounding boxes in evaluation frames.
# Additionally, calculate and append useful parameters using output satellite telemetry, range measurements and object detection predictions.

def dict2vec(dict):
    x = dict['x']
    y = dict['y']
    z = dict['z']
    return np.array((x,y,z))

def extract_YOLO_annotations(annotationPath, annotationList, frame):
    with open(os.path.join(annotationPath, annotationList)) as f:
        annotations = f.read()
    annotations = list(map(float, annotations.split(' ')))

    image_size = frame.shape[0:2]
    x_min = round((annotations[1] - annotations[3] / 2) * image_size[0])
    y_min = round((1 - (annotations[2] - annotations[4] / 2)) * image_size[1])
    x_max = round(x_min + (annotations[3] * image_size[0]))
    y_max = round(y_min - (annotations[4] * image_size[1]))
    return (x_min,y_min,x_max,y_max)

def extract_JSON_annotations(annotationPath, epoch):
    # Read JSON file
    with open(os.path.join(annotationPath, trueAnnot[0])) as json_file:
        observations = json.load(json_file)
    # Read absolute bounding box boundaries
    x = observations['bbox'][epoch]['Xcenter']
    y = observations['bbox'][epoch]['Ycenter']
    w = observations['bbox'][epoch]['width']
    h = observations['bbox'][epoch]['height']
    # Read satellite-debris direction vector
    sd = dict2vec(observations['telemetry'][epoch]['directionVector_actual'])
    # Read satellite position vector
    es = dict2vec(observations['telemetry'][epoch]['satellitePosition'])
    # Read satellite speed
    Sspeed = observations['telemetry'][epoch]['SatelliteSpeed']
    return (x,y,w,h, sd,es,Sspeed)

def process_readings(sd,sdPrev,es,esPrev,Nframes,focal_length,w,sensorSz,size):
    # Calculate satellite-debris range
    SD = np.linalg.norm(sd)
    # Calculate debris speed
    ed = np.add(es, sd) # debris position vector in ECI coordinate frame
    edPrev = np.add(esPrev, sdPrev)
    d = ed - edPrev # debris direction vector between current and previous states
    D = np.linalg.norm(d) # debris distance travelled between states (assuming straight line path)
    t = Nframes/fps
    DebSpeed = D/t
    # Calculate debris size
    size_on_sensor_mm = sensorSz*(w*size[0])/size[0]
    DebSize = SD*size_on_sensor_mm/focal_length # [m]
    return (SD, DebSpeed, DebSize, ed)

def get_indexes_max_value(l):
    max_value = max(l)
    if l.count(max_value) > 1:
        return [i for i, x in enumerate(l) if x == max(l)]
    else:
        return l.index(max(l))

def draw_label(frame, text, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 255, 255)
    thickness = cv2.FILLED
    margin = -5

    text = text.split('\n')
    MAXindex = get_indexes_max_value(text)
    max_txt_size = cv2.getTextSize(text[MAXindex], font_face, scale, thickness)

    end_x = pos[0] + max_txt_size[0][0] + margin
    end_y = pos[1] - max_txt_size[0][1] - margin
    for i, line in enumerate(text):
        y = pos[1] + i * end_y
        cv2.putText(frame, line, (pos[0], y), font_face, scale, color, 1, cv2.LINE_AA)


# Camera properties
fps = 60
focal_length = 2680 # focal length [mm]
sensorSz = 26.624000549316406 # square sensor size [mm]

OSF = 40 # specify OffSet Factor for a more visible bbox [in pixels]
videoFolder = '/data/final_system_animation'
files = os.listdir(videoFolder)

# Separate true label files from animations (true labels = actual object location)
animations = []
trueAnnot = []
for file in files:
    if file.endswith('.avi'):
        animations.append(file)
    elif file.endswith('.txt'):
        trueAnnot.append(file)

# Extract frames from individual detection video and superimpose bounding box from annotation data
epoch = 0
frames = []
text1 = 'FPS: \nrange:  km\ndetected position (ECI): m\ndetected size:  m\ndetected speed:  m/s\nsatellite speed:  m/s'
text2 = 'detected position (ECI): m\n'
for vid in animations:
    vidcap = cv2.VideoCapture(os.path.join(videoFolder, vid))
    Nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for n in range(Nframes):
        success, frame = vidcap.read()
        size = frame.shape[0:2]

        if n == Nframes-1:
            # Read true annotations
            x, y, w, h, sd, es, SatSpeed = extract_JSON_annotations(videoFolder, epoch)
            # Convert to bounding box coordinates
            x_min = round((x - w / 2) * size[0])
            x_max = round((x + w / 2) * size[0])
            y_min = round((y - h / 2) * size[1])
            y_max = round((y + h / 2) * size[1])
            # Draw bounding box
            cv2.rectangle(frame,(x_min-OSF,y_min-OSF),(x_max+OSF,y_max+OSF),(0,255,0),3) # add rectangle to image
            if epoch>0:
                # Process telemetry/detection data to output useful parameters
                SD, DebSpeed, DebSize, ed = process_readings(sd,sdPrev,es,esPrev,Nframes,focal_length,w,sensorSz,size)
                # Update parameters textbox
                # https://stackoverflow.com/questions/54607447/opencv-how-to-overlay-text-on-video
                text1 = 'FPS: {}\nrange: {} km\ndetected size: {} m\ndetected speed: {} m/s\nsatellite speed: {} m/s'.format(
                    fps, round(SD/1000,2), round(DebSize,1), round(DebSpeed), round(SatSpeed))
                text2 = 'detected position (ECI): {} m\n'.format(np.round(ed,1))
            # Assign current processed parameters as "previous" for next iteration
            sdPrev = sd
            esPrev = es
        # Append textbox in frame
        draw_label(frame, text1, (size[0]-480, 50))
        draw_label(frame, text2, (round(size[0]/4), size[1]-30))
        frames.append(frame)
    epoch += 1
# Write final video combining all individual animations
video = cv2.VideoWriter(os.path.join(videoFolder, 'system_example.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frames)):
    video.write(frames[i])
video.release()
