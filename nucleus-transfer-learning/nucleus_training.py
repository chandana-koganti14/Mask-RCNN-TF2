import os
from numpy import zeros, asarray
import pandas as pd
import numpy as np
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import skimage
from skimage.io import imread
import csv
import scipy
import matplotlib.pyplot as plt
import cv2
from mrcnn.model import MaskRCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class NucleusDataset(mrcnn.utils.Dataset):

    
    def load_dataset(self, dataset_dir, is_train=True): 
        self.add_class("nucleus", 1, "tumor")
        self.add_class("nucleus", 2, "fibroblast")
        self.add_class("nucleus", 3, "lymphocyte")
        self.add_class("nucleus", 4, "plasma_cell")
        self.add_class("nucleus", 5, "macrophage")
        self.add_class("nucleus", 6, "mitotic_figure")
        self.add_class("nucleus", 7, "vascular_endothelium")
        self.add_class("nucleus", 8, "myoepithelium")
        self.add_class("nucleus", 9, "apoptotic_body")
        self.add_class("nucleus", 10, "neutrophil")
        self.add_class("nucleus", 11, "ductal_epithelium")
        self.add_class("nucleus", 12, "eosinophil")
        self.add_class("nucleus", 13, "unlabeled")
        

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.csv'
           
            self.add_image(source="nucleus", image_id=image_id, path=img_path, annot_path=ann_path, class_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return imread(info['path'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    from skimage.io import imread
    import skimage.color
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annot_path']
        image = self.load_image(image_id)
        boxes, w, h = self.extract_boxes(path,image)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []

        class_label_mapping = {
            'tumor': 1,
            'fibroblast': 2,
            'lymphocyte': 3,
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

        for i, box in enumerate(boxes):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            class_id = class_label_mapping.get(box[4], 0)
            masks[row_s:row_e, col_s:col_e, i] = class_id
            class_ids.append(class_label_mapping.get(box[4], 0))
        return masks, asarray(class_ids, dtype='int32')

    import csv

    def extract_boxes(self, filename,image):
      boxes = []
      df = pd.read_csv(filename)
      for _, row in df.iterrows():
          raw = str(row['raw_classification'])
          xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
          coors = [xmin, ymin, xmax, ymax, raw]
          boxes.append(coors)
      return boxes, image.shape[1], image.shape[0]
    def get_class_counts(self):
        class_counts = {class_info['id']: 0 for class_info in self.class_info}
        for image_id in self.image_ids:
            _, class_ids = self.load_mask(image_id)
            for class_id in class_ids:
                class_counts[class_id] += 1
        return class_counts
    def compute_class_weights_per_image(self, class_ids):
        unique_classes, class_counts = np.unique(class_ids, return_counts=True)
        total = sum(class_counts)
        class_weights = {class_id: total / count if count > 0 else 0 for class_id, count in zip(unique_classes, class_counts)}
        return class_weights

    def compute_class_weights(self):
        class_weights_per_image = []
        for image_id in self.image_ids:
            _, class_ids = self.load_mask(image_id)
            class_weights = self.compute_class_weights_per_image(class_ids)
            class_weights_per_image.append(class_weights)
        return class_weights_per_image



class NucleusConfig(mrcnn.config.Config):
    NAME = "nucleus_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 14

    STEPS_PER_EPOCH = 10
    def __init__(self, class_weights_per_image=None):
        super().__init__()
        self.CLASS_WEIGHTS_PER_IMAGE = class_weights_per_image
train_dataset = NucleusDataset()
train_dataset.load_dataset(dataset_dir='nucleus', is_train=True)
train_dataset.prepare()
validation_dataset = NucleusDataset()
validation_dataset.load_dataset(dataset_dir='nucleus', is_train=False)
validation_dataset.prepare()
class_counts = train_dataset.get_class_counts()
class_weights_per_image = train_dataset.compute_class_weights()

class_names = train_dataset.class_names
class_labels = [class_names[class_id] for class_id in class_counts.keys()]
nucleus_config = NucleusConfig(class_weights_per_image=class_weights_per_image)
class_weights = nucleus_config.CLASS_WEIGHTS_PER_IMAGE
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=nucleus_config)
model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
#print("Class weights used during training:", nucleus_config.CLASS_WEIGHTS_PER_IMAGE)
model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset,
            learning_rate=nucleus_config.LEARNING_RATE, 
            epochs=2, 
            layers='heads')


model_path = 'Nucleus_MaskRCNN_50epochs.h5'
model.keras_model.save_weights(model_path)
