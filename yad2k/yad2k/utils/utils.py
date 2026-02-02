"""Miscellaneous utility functions for YOLO."""

from functools import reduce
import numpy as np
import imghdr
from PIL import Image, ImageDraw


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right."""
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def read_classes(classes_path):
    """Loads class names from a file"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    """Loads anchor boxes from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def scale_boxes(boxes, image_shape):
    """Scales predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = np.array([height, width, height, width])
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def preprocess_image(img_path, model_image_size):
    """Preprocess image for model prediction"""
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    return image, image_data
