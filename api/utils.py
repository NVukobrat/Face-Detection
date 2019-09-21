import tensorflow as tf
import numpy as np
import cv2


def yolo_boxes(pred, anchors, classes):
    """
    Yolo boxes uses network output and transforms it to
    obtaint bounding box prediction. Formulas are as following:

    bx = sig(tx) + cx
    by = sig(ty) + cy
    bw = pw * e^tw
    bh = ph * e^th

    Where bx, by, bw and bh represent coordinates and dimensions
    of the final prediction. Following tx, ty, tw and th is what the network
    outputs. Then, cx and cy are top left coordinated of the grid. And lastly,
    pw and ph represent anchors dimensions for the box. Sigmoid functions are
    used in order to keep center coordinates prediction within the borders of
    the bounding box.

    Arguments:
        pred:
        anchors:
        classes:

    Returns:
        bbox: Bounding box.
        objectness: Probability [0, 1] that an object is contained within
        a bounding box.
        class_probs: Confidence score for each class (using sigmoid function,
        not mutual exclusive objects).
        pred_box: Original coordinates and dimensions (x, y, w, h) of
        the predictions for loss.

    """
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    """
    Non-maximum Suppression solves a problem with multiple detections
    in the same region. Namely, multiple grid could detect same objects. So,
    NMS is used to remove thoes multiple detections.

    Arguments:
        outputs:
        anchors:
        masks:
        classes:

    Returns:
        boxes:
        scores:
        classes:
        valid_detections:

    """
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255

    return x_train


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), x1y1,
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    return img
