import os, cv2
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor


# References
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
# https://detectron2.readthedocs.io/tutorials/datasets.html
# https://blog.roboflow.ai/how-to-train-detectron2/

train_data_path = '/user/HS122/ml00503/dissertation/data/Faster_R-CNN/train'
test_data_path = '/user/HS122/ml00503/dissertation/data/Faster_R-CNN/test'
data_paths = [train_data_path, test_data_path]

# Parse input data to Detectron2 format
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

datasets = ['train', 'test']
for d in range(0,len(data_paths)):
    DatasetCatalog.register('debris_dataset_'+datasets[d], lambda d=d: get_dataset(data_paths[d]))
    MetadataCatalog.get('debris_dataset_' + datasets[d]).set(thing_classes=['debris']) # assign object name to label


# VISUALIZE LOADED DATA #
import random
from detectron2.utils.visualizer import Visualizer

metadata = MetadataCatalog.get('debris_dataset_train')
dataset_dicts = get_dataset(data_paths[0])
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.3)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('img{}'.format(d), out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Setup cfg parameters#
# Use Faster-RCNN 101 layer model with ResNet backbone (R101-C4)
cfg = get_cfg()
model_zoo.get('COCO-Detection/faster_rcnn_R_50_C4_3x.yaml', True) # True: Lets training initialize from model #faster_rcnn_R_101_C4_3x.yaml
cfg.DATASETS.TRAIN = ('debris_dataset_train',)
cfg.DATASETS.TEST = ('debris_dataset_test',)

cfg.DATALOADER.NUM_WORKERS = 4
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_C4_3x.yaml')
cfg.SOLVER.IMS_PER_BATCH = 2 # CUDA out of memory if > 1
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 80000 #(1000)
cfg.SOLVER.MAX_ITER = 100000 # maximum number of training iterations (1500)
cfg.SOLVER.STEPS = (80000, 100000) # (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.CHECKPOINT_PERIOD = 25000 # make a checkpoint .weights file every 'x' iterations

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # specify number of classes
#cfg.TEST.EVAL_PERIOD = 2000 # evaluate every 500 iterations


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Modify train/test dataloaders to remove default augmentations
#class Trainer(DefaultTrainer):
#    @classmethod
#    def build_test_loader(cls, cfg, dataset_name):
#        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False, augmentations=[]))
#
#    @classmethod
#    def build_train_loader(cls, cfg):
#        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations=[]))

if __name__ == '__main__':
    # Training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference and Evaluation
    from detectron2.data import  build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = ('debris_dataset_test',)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator('debris_dataset_test', cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "debris_dataset_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    from detectron2.utils.visualizer import Visualizer, ColorMode
    import random

    test_metadata = MetadataCatalog.get('debris_dataset_train')
    dataset_dicts = get_dataset(test_data_path)
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d['file_name'])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_metadata,
                       scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('img',  out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
