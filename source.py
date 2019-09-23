import time

import cv2
import tensorflow as tf

from api.config import (
    WEIGHT_PATH,
    CLASS_NAME_PATH,
    TRANS_IMG_SHAPE,
)
from api.model import yolo_v3
from api.utils import transform_images, draw_outputs


def main():
    # Create model
    yolo = yolo_v3()

    # Load model weights
    yolo.load_weights(WEIGHT_PATH)

    # Load class names
    class_names = [c.strip() for c in open(CLASS_NAME_PATH).readlines()]

    # Load image from the camera
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Image from numpy to byte
        start = time.time()
        success, encoded_image = cv2.imencode('.jpg', frame)
        image = encoded_image.tobytes()
        print("Image from numpy to byte {0}".format(time.time() - start))

        # Read image
        start = time.time()
        image = tf.image.decode_image(image, channels=3)
        image = tf.expand_dims(image, 0)
        image = transform_images(image, TRANS_IMG_SHAPE)
        print("Read image {0}".format(time.time() - start))

        # Run model
        start = time.time()
        boxes, scores, classes, nums = yolo(image)
        print("Run model {0}".format(time.time() - start))

        # Draw outputs
        start = time.time()
        image = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
        print("Draw outputs {0}".format(time.time() - start))

        # Display the resulting frame
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
