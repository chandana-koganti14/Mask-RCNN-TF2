import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import sklearn
import numpy as np
from sklearn.metrics import classification_report
CLASS_NAMES = ["BG","tumor", "fibroblast", "lymphocyte", "plasma_cell", 
               "macrophage", "mitotic_figure", "vascular_endothelium",  
               "myoepithelium", "apoptotic_body", "neutrophil", 
               "ductal_epithelium", "eosinophil", "unlabeled"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())
model.load_weights(filepath="Nucleus_MaskRCNN_50epochs.h5", 
                   by_name=True)
image = cv2.imread(r"C:\Users\ADMIN\Mask-RCNN-TF2\nucleus-transfer-learning\nucleus\images\TCGA-AR-A0U4-DX1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r1 = model.detect([image], verbose=0)
r1 = r1[0]
CLASS_NAME_TO_ID = {
  "tumor": 1,
  "fibroblast": 2, 
  "lymphocyte": 3,
  'plasma_cell': 4,
            'macrophage': 5,
            'mitotic_figure': 6,
            'vascular_endothelium': 7,
            'myoepithelium': 8,
            'apoptotic_body': 9,
            'neutrophil': 10,
            'ductal_epithelium': 11,
            'eosinophil': 12,
            'unlabeled': 13
}
GT_CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
CSV_FILE = r"C:\Users\ADMIN\Mask-RCNN-TF2\nucleus-transfer-learning\nucleus\annots\TCGA-AR-A0U4-DX1.csv"
gt_class_names = []
gt_bboxes = []

with open(CSV_FILE) as f:
    reader = csv.reader(f)
    next(reader) 
    df = pd.read_csv(CSV_FILE)
    for _, row in df.iterrows():
        raw = str(row['raw_classification'])
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        gt_class_names.append(raw)  
        gt_bboxes.append([xmin, ymin, xmax, ymax])
    unique_gt_classes = np.unique(gt_class_names)   
gt_class_ids = [GT_CLASS_NAME_TO_ID[name] for name in gt_class_names]
y_true_class = gt_class_ids
y_pred_class = r1['class_ids']
y_true_bbox = gt_bboxes 
y_pred_bbox = r1['rois']
pred_class_ids = r1['class_ids']

if len(y_true_class) > len(y_pred_class):
  y_pred_class = np.pad(y_pred_class, (0, len(y_true_class)-len(y_pred_class)), mode='constant')
elif len(y_true_class) < len(y_pred_class):
  y_true_class = np.pad(y_true_class, (0, len(y_pred_class)-len(y_true_class)), mode='constant') 

accuracy = sklearn.metrics.accuracy_score(y_true_class, y_pred_class)
precisions=[]
recalls=[]
ious = []
f1s = []
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2
    x1_i = max(x1, x1_b)
    y1_i = max(y1, y1_b)
    x2_i = min(x2, x2_b)
    y2_i = min(y2, y2_b)
    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area
    return iou
print("Unique GT Classes:", unique_gt_classes)
print("Ground Truth Classes:", y_true_class)
print("Predicted classes",pred_class_ids)
unique_gt_classes = np.unique(y_true_class)
tp = 0
fp = 0
fn = 0

IOU_THRESHOLD = 0.2
num_predictions = len(y_pred_bbox)
print("Num Predictions:", num_predictions)

precisions = [] 
recalls = []
ious = []
f1s = []

for i in range(num_predictions):
    pred_box = y_pred_bbox[i]
    pred_class_id = y_pred_class[i]
    
    iou_max = 0
    match_gt_id = None
    for j in range(len(y_true_bbox)):
        iou = calculate_iou(pred_box, y_true_bbox[j]) 
        if iou > iou_max:
            match_gt_id = j
            iou_max = iou
    if iou_max >= IOU_THRESHOLD:
        if pred_class_id == y_true_class[match_gt_id]:
            tp += 1  
        else:
            fp += 1
    else: 
        fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = iou_max
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    precisions.append(precision)
    recalls.append(recall)
    ious.append(iou)
    f1s.append(f1)
avg_precision = np.nanmean(precisions) if len(precisions) > 0 else 0.0
avg_recall = np.nanmean(recalls) if len(recalls) > 0 else 0.0
avg_iou = np.nanmean(ious) if len(ious) > 0 else 0.0
avg_f1 = np.nanmean(f1s) if len(f1s) > 0 else 0.0
print("Avg Precision:", avg_precision)
print("Avg Recall:", avg_recall)
print("Avg IOU:", avg_iou)
print("Avg F1 Score:", avg_f1)
print("Accuracy: ", accuracy)


mrcnn.visualize.display_instances(image=image,
                                  boxes=r1['rois'], 
                                  masks=r1['masks'], 
                                  class_ids=r1['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r1['scores'])