import os, cv2

# Description:
# Specify desired animation and draw bounding box of final frame from extracted label data (label.txt file).
# Yolo format = label, x_min, y_min, width, height
#
# Inputs:
# 1. filepath: path of animation directory.
# 2. animation_name: name of blended animation (.png) file.


filepath = '/data/batch1_R1_LEO_8frames'
animation_name = 'animation2_R1.0_S6900ms_orbitDiff_38345m_H_Flip'
annotation_file = animation_name+'.txt'
image_file = animation_name+'.png'

img = cv2.imread(os.path.join(filepath, image_file))
image_size = img.shape

with open(os.path.join(filepath, annotation_file)) as f:
    annotations = f.read()
annotations = list(map(float, annotations.split(' ')))

label = annotations[0]
# Convert normalized to pixels
# Blender's position offset is at bottom left, opencv's is at top left (hence 1-y)
x_min = round((annotations[1]-annotations[3]/2) * image_size[0])
y_min = round((1-(annotations[2]-annotations[4]/2)) * image_size[1])
x_max = round(x_min + (annotations[3] * image_size[0]))
y_max = round(y_min - (annotations[4] * image_size[1]))

bounded_img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,255,0),1) # add rectangle to image
imgS = cv2.resize(bounded_img, (960, 540))
cv2.imshow('bounded_img', imgS)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('bounded_{}.png'.format(animation_name), bounded_img)