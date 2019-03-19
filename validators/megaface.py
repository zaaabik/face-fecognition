import errno
import json
import optparse
import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from metric_learning.main import face_align
from metric_learning.resnet34 import Resnet34
from validators.matio import save_mat


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main():
    resnet = Resnet34(0, 0, 128, 128, arch='test3')
    resnet = resnet.create_model()
    resnet.load_weights(weights, by_name=True)
    with open(images_paths) as file:
        feature_file = json.load(file)
        paths = feature_file['path']
        images = []
        for path in paths:
            image_path = os.path.join(base_path, path)
            image = imread(image_path)
            try:
                image = face_align(image)
                image = np.array(resize(image, (128, 128)))
                images.append(image)
            except:
                print(image_path)
                exit(0)
        embedding = resnet.predict(np.array(images))
        for idx, path in enumerate(paths):
            out_path = os.path.join(out, path)
            out_path = out_path + '.weights'
            head, _ = os.path.split(out_path)
            mkdir_p(head)
            save_mat(out_path, embedding[idx])


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--images_paths', type='string')
    parser.add_option('--base_path', type='string')
    parser.add_option('--weights', type='string')
    parser.add_option('--out', type='string')
    options, _ = parser.parse_args()
    images_paths = options.images_paths
    base_path = options.base_path
    weights = options.weights
    out = options.out
    main()
