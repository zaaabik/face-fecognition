import errno
import json
import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_images(image_paths, size):
    images = np.array([
        get_image(file_name, size)
        for file_name in image_paths])
    return images


def get_image(image_path, size):
    image = resize(imread(image_path), (size, size), anti_aliasing=True, mode='reflect') - 0.5
    image *= 2
    return image


def parse_landmarks(path):
    with open(path) as file:
        feature_file = json.load(file)
        feature_file = feature_file['landmarks']

        right_eye = feature_file.get('0', None)
        if right_eye is None:
            raise AttributeError
        right_eye = right_eye['x'], right_eye['y']
        left_eye = feature_file.get('1', None)
        if left_eye is None:
            raise AttributeError
        left_eye = left_eye['x'], left_eye['y']
    return np.array(right_eye), np.array(left_eye)
