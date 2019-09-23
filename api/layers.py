import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Add,
    Input,
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    UpSampling2D,
    Concatenate,
    Lambda
)
from tensorflow.python.keras.regularizers import l2 as L2


#################
# ### Darknet ###
#################

def darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32, 3)
    x = darknet_block(x, 64, 1)
    x = darknet_block(x, 128, 2)  # skip connection
    x = x_36 = darknet_block(x, 256, 8)  # skip connection
    x = x_61 = darknet_block(x, 512, 8)
    x = darknet_block(x, 1024, 4)

    return Model(inputs, (x_36, x_61, x), name=name)


def darknet_conv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=L2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def darknet_block(x, filters, blocks):
    x = darknet_conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknet_residual(x, filters)
    return x


def darknet_residual(x, filters):
    prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    x = Add()([prev, x])
    return x


##############
# ### Yolo ###
##############

def yolo_conv(filters, name=None):
    def yolo_conv_inner(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv_inner


def yolo_output(filters, anchors, classes, name=None):
    """
    Output of the YOLO is feature map since 1x1 convolutionals
    are used instead of classifier or regressor.

    Feature map consists of (B x (5 + C)) entries. B represent
    the number of bounding boxes that each cell could predict.
    Each of the bounding boxes has 5 + C attributes. The 5 attributes
    represent: center coordinates (x, y), dimensions (w, h) and
    objectness score. Furthermore, C represent class confidence for
    each bounding box. The expectation is that each cell predict an
    object withing coresponding bounding box.

    Arguments:
        filters: Number of convolutional filters.
        anchors: Number of anchors that will be used.
        classes: Number of existing classes.
        name: Custom name of the network segment.

    Returns:
        Yolo output model.
    """

    def yolo_output_inner(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)

        return Model(inputs, x, name=name)(x_in)

    return yolo_output_inner
