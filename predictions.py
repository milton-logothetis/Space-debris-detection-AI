import os, cv2, random
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

def get_dataset(data_path):
    data = [file for file in os.listdir(data_path) if file.endswith('.png')]
    annotations = [file for file in os.listdir(data_path) if file.endswith('.txt')]
    list = []
    for i in range(0, len(data)):
        filename = data[i][0:-4]
        imgpath = os.path.join(data_path, data[i])
        height, width = cv2.imread(imgpath).shape[:2]
        f = open(os.path.join(data_path, filename+'.txt'))
        [label, x, y, bbwidth, bbheight] = f.read()[0:-1].split(' ')
        # Convert to absolute coordinates(0,0 at top left)
        x = (float(x)-float(bbwidth)/2)*width
        y = (1-float(y)-float(bbheight)/2)*height
        bbwidth = float(bbwidth)*width
        bbheight = float(bbheight)*height
        dict = {
            'file_name': str(os.path.join(data_path, data[i])),
            'height': height,
            'width': width,
            'image_id': i,
            'annotations': [{
                'bbox': [x,y,bbwidth,bbheight],
                'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': int(label)
                }]
             }
        list.append(dict)
    return list

# Specify paths to train/test data and output directories
train_data_path = '/train'
test_data_path = '/test'
out_path = '/results'
data_paths = [train_data_path, test_data_path]

datasets = ['train', 'test']
for d in range(0,len(data_paths)):
    DatasetCatalog.register('debris_dataset_'+datasets[d], lambda d=d: get_dataset(data_paths[d]))
    MetadataCatalog.get('debris_dataset_' + datasets[d]).set(thing_classes=['debris']) # assign object name to label

cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # specify number of classes
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold for this model
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.DATASETS.TRAIN = ('debris_dataset_train',)
cfg.DATASETS.TEST = ('debris_dataset_test',)

predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get('debris_dataset_test')
dataset_dicts = get_dataset(test_data_path)
bbox = []
for d in random.sample(dataset_dicts, 100):
	im = cv2.imread(d['file_name'])
	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1],metadata=test_metadata,scale=1)# Inference and Evaluation
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	bbox.append(outputs["instances"].pred_boxes)
	#cv2.imshow('img',  out.get_image()[:, :, ::-1])
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	cv2.imwrite(os.path.join(out_path, d['file_name'].split('/')[-1]), out.get_image()[:, :, ::-1]) # FIX error which saves & overwrites images in test_data_path
#with open(os.path.join(outpath, 
	
