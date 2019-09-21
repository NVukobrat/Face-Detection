import numpy as np

###########################
# ### General constants ###
###########################

# Image shape (Should be single int, ex: 32 → 32x32)
IMG_SHAPE = None

# Number of channels of the dataset and generator output images.
N_CHANNELS = 3

# Number of classes
N_CLASSES = 80

#################################
# ### Model related constants ###
#################################

# Transform image size
TRANS_IMG_SHAPE = 416

# Path to the model weights
WEIGHT_PATH = "checkpoints/yolov3.tf"

# Path to file that contain class names
CLASS_NAME_PATH = "data/coco.names"

# Yolo anchors
YOLO_ANCHORS = np.array([
    (10, 13),
    (16, 30),
    (33, 23),
    (30, 61),
    (62, 45),
    (59, 119),
    (116, 90),
    (156, 198),
    (373, 326)
], np.float32) / TRANS_IMG_SHAPE

# Yolo anchor masks
YOLO_ANCHOR_MASKS = np.array([
    [6, 7, 8],
    [3, 4, 5],
    [0, 1, 2]
])
