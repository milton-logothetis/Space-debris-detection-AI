import os, json, cv2
import numpy as np
import datetime
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter2D
from math import *

# https://docs.poliastro.space/en/stable/user_guide.html#defining-the-orbit-orbit-objects
# https://docs.poliastro.space/en/stable/api/safe/twobody/twobody_index.html
# https://space.stackexchange.com/questions/1904/how-to-programmatically-calculate-orbital-elements-using-position-velocity-vecto

# DESCRIPTION:
# Reads the data exported from the satellite detection system, such as observing satellite telemetry and bounding box
# predictions in order to determine the osculating or instaneous Keplerian orbit of the detected object and plot it.
# Note the output plot visualizes the calculated orbit plane, i.e in the perifocal frame.

# WORKFLOW:
# For epoch0:
# 1. Read directionVector (SD0) and use Pythagoras to calc distance (d0)
# 2. Record satellitePosition (SP0) = directionVector from Earth to satellite
# 3. Calculate directionVector from Earth to debris (ED0)
# For epoch>0:
# 1. Read SD1 and calc d1
# 2. Record SP1, calc ES1
# 3. Calculate ED1
# 4. From camera FPS and number of frames between epochs0-1 calc time interval between two states
# 5. Use epochs0-1 to calculate distance travelled and divide by time to get velocity vector
# 6. Repeat for consecutive epochs

def dict2vec(dict, epoch):
    x = dict[epoch]['x']
    y = dict[epoch]['y']
    z = dict[epoch]['z']
    return np.array((x,y,z))

datapath ='../data/OD_test1'
datadir = os.listdir(datapath)
num_epochs = len(datadir)-2

# Count number of sequences (IMPROVE: output frames, FPS in JSON file from Blender)
fps = 60
sequences = []
frames = []
count = 0
for file in datadir:
    if file.endswith('.avi'):
        sequences.append(file)
        # Extract number of frames
        vidcap = cv2.VideoCapture(os.path.join(datapath, sequences[count]))
        frames.append(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
        vidcap.release()
        count += 1
# Read observations/predictions from output JSON file
with open(os.path.join(datapath, 'annotations0.txt')) as json_file:
    observations = json.load(json_file)

sd_actual = []
sd_pred = []
es = []
r_actual = []
r_pred = []
ED_actual = []
ED_pred = []
v_actual = []
v_pred = []
ranges = []
speed_actual = []
speed_pred = []
orb = []
op = OrbitPlotter2D() # Initialiaze orbit plot
for epoch in range(num_epochs):
# Satellite-debris direction vectors
    sd_actual.append(observations['telemetry'][epoch]['directionVector_actual'])              # Actual satellite-debris direction vector
    # Calculate predicted satellite-debris vector
    K = np.array(observations['CameraCalibration'][epoch]['intrinsics']) # camera intrinsics matrix
    RT = np.array(observations['CameraCalibration'][epoch]['extrinsics']) # camera extrinsics matrix
    # Extract bounding box predictions
    x = observations['bbox'][epoch]['Xcenter']
    y = observations['bbox'][epoch]['Ycenter']
    p = np.array([[x],[y],[1]]) # convert location of bbox center to homogenous vector
    uv = np.dot(np.linalg.inv(K),p)# unit vector in camera coordinate frame
    uv = uv/np.linalg.norm(uv) # normalize unit vector
    # Simulated range measurement (This step would be measured from instrument data in real scenario)
    ranges.append(sqrt(sd_actual[epoch]['x']**2+sd_actual[epoch]['y']**2+sd_actual[epoch]['z']**2))  # Satellite-debris distance
    sdcam = uv*ranges[epoch] # satellite-debris direction vector in camera coordinate frame
    sd_pred.append(np.dot(sdcam.transpose(),RT)) # Predicted satellite-debris direction vector in world coordinate frame
# Calculate debris distance from Earth
    es.append(observations['telemetry'][epoch]['satellitePosition'])            # Satellite position vector
    r_actual.append(np.add(dict2vec(es, epoch), dict2vec(sd_actual, epoch)))                  # Debris position vector
    r_pred.append(np.add(dict2vec(es, epoch), sd_pred[epoch][0,[0,1,2]]))
#    ED_actual.append(np.linalg.norm(r_actual[epoch]))                                # Earth-debris distance
#    ED_pred.append(np.linalg.norm(r_pred[epoch]))

    if epoch > 0:
    # Debris orbit calculations
        # Calculate debris direction vector between last and previous state
        d_actual = r_actual[epoch] - r_actual[epoch-1]
        d_pred = r_pred[epoch] - r_pred[epoch-1]
        # Calculate distance travelled by debris (Assuming straight line path)
        D_actual = np.linalg.norm(d_actual)
        D_pred = np.linalg.norm(d_pred)
        # Calculating time between epochs (IMPROVE: extract from Blender simulation)
        t = (frames[epoch]-1) / fps
        # Calculate speed
        #speed_actual.append(observations['telemetry'][epoch]['cameraSpeed'])
        speed_actual.append(D_actual/t)
        speed_pred.append(D_pred/t)
        # Calculate velocity vector
        v_actual.append(speed_actual[epoch-1]/D_actual * d_actual)
        v_pred.append(speed_pred[epoch-1]/D_pred * d_pred)

# Average observations
# Actual
ravg_actual = np.mean(r_actual, axis=0)
vavg_actual = np.mean(v_actual, axis=0)
# Define and plot orbit
orb = Orbit.from_vectors(Earth, ravg_actual * u.m, vavg_actual * u.m / u.s)
#orb = orb.propagate(60 * u.min) # propagate orbit by 1-hr
p = orb.plot(label='Space debris orbit')
p[0].set_color([128/255,0,128/255]) # set orbit color
p[1].set_color([128/255,0,128/255]) # set debris color

# Predicted
ravg_pred = np.mean(r_pred, axis=0)
vavg_pred = np.mean(v_pred, axis=0)
# Define and plot orbit
orb = Orbit.from_vectors(Earth, ravg_pred * u.m, vavg_pred * u.m / u.s)
p = orb.plot(label='Space debris predicted')
p[0].set_color([128/255,0,128/255]) # set orbit color
p[1].set_color([128/255,0,128/255]) # set debris color

#op.plot(orb[0], label='Debris orbit')
#op.plot(orb[1], label='Debris orbit2')

# Alternative calculation of distance travelled using Trigonometry
# Calculate angle between previous and current Earth-debris (ed) vectors.
#theta = acos((np.dot(ed[epoch], ed[epoch - 1])) / (ED[epoch] * ED[epoch - 1]))
# Applying cosine rule:
#D = sqrt(ED[epoch] ** 2 + ED[epoch - 1] ** 2 - 2 * ED[epoch] * ED[epoch - 1] * cos(theta))  # Distance travelled by debris between epochs (assuming straight line)



