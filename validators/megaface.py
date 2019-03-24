import errno
import json
import optparse
import os

import cv2
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
            image = np.array(resize(image, (128, 128)))
            images.append(image)

        embedding = resnet.predict(np.array(images))
        for idx, path in enumerate(paths):
            out_path = os.path.join(out, path)
            out_path = out_path + '.weights'
            head, _ = os.path.split(out_path)
            mkdir_p(head)
            save_mat(out_path, embedding[idx])


def align():
    with open(images_paths) as file:
        feature_file = json.load(file)
        paths = feature_file['path']
        for idx, path in enumerate(paths):
            if idx < skip:
                continue
            image_path = os.path.join(base_path, path)
            image = cv2.imread(image_path)
            try:
                image = face_align(image)
            except:
                print(f'no face in {image_path}')

            out_path = os.path.join(out, path)
            head, file = os.path.split(out_path)
            is_exists = os.path.exists(head)
            try:
                if not is_exists:
                    mkdir_p(head)
                if file.find('.') == -1:
                    out_path += '.jpg'
                    cv2.imwrite(out_path, image)
                    os.rename(out_path, out_path[:-4])
                elif file[:-4] == '.gif':
                    out_path = out_path[:-4] + '.jpg'
                    cv2.imwrite(out_path, image)
                    os.rename(out_path, out_path[:-4] + '.gif')
                else:
                    cv2.imwrite(out_path, image)

            except Exception as e:
                print(e)
                print(f'can`t write image {out_path}')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--images_paths', type='string')
    parser.add_option('--base_path', type='string')
    parser.add_option('--weights', type='string')
    parser.add_option('--out', type='string')
    parser.add_option('--mode', type='string', default='full')
    parser.add_option('--skip', type='int', default=0)
    options, _ = parser.parse_args()
    images_paths = options.images_paths
    base_path = options.base_path
    weights = options.weights
    skip = options.skip
    out = options.out
    mode = options.mode
    if mode == 'create':
        main()
    elif mode == 'align':
        align()
