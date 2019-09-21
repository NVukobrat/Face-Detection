from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Input,
    Lambda
)

from api.config import (
    YOLO_ANCHORS,
    YOLO_ANCHOR_MASKS,
    N_CHANNELS,
    N_CLASSES,
    IMG_SHAPE
)
from api.layers import (
    darknet,
    yolo_conv,
    yolo_output,
)
from api.utils import (
    yolo_boxes,
    yolo_nms
)


def yolo_v3(size=IMG_SHAPE, channels=N_CHANNELS, anchors=YOLO_ANCHORS, masks=YOLO_ANCHOR_MASKS, classes=N_CLASSES,
            training=False):
    """
    In the problem of object detections, most common approach is to use
    convolutional layers to learn the features, and then pass thoes to the
    classifier or regressor which makes prediction/detection. On the other side
    in YOLO, prediction is done using 1x1 convolutional layers.

    Arguments:
        size:
        channels:
        anchors:
        masks:
        classes:
        training:

    Returns:
        Yolo network model.
    """
    x = inputs = Input([size, size, channels])

    x_36, x_61, x = darknet(name='yolo_darknet')(x)

    x = yolo_conv(512, name='yolo_conv_0')(x)
    output_0 = yolo_output(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = yolo_conv(256, name='yolo_conv_1')((x, x_61))
    output_1 = yolo_output(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = yolo_conv(128, name='yolo_conv_2')((x, x_36))
    output_2 = yolo_output(128, len(masks[2]), classes, name='yolo_output_2')(x)

    # if training:
    #     return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')
