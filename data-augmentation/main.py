import optparse
import os
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import dlib
import numpy as np
from PIL import Image
from PIL import ImageFilter
from imutils import face_utils

from helpers.helpers import mkdir_p

parser = optparse.OptionParser()
parser.add_option("-i", "--input")
parser.add_option("-o", "--output", type='string')
parser.add_option("-c", "--cpu", type='int', default=4)
(options, args) = parser.parse_args()
input_folder = options.input
output_folder = options.output
cpu = options.cpu

sep = os.path.sep

glass_prefix = '_glass_'

detector = dlib.get_frontal_face_detector()
predictor_data_path = '../shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_data_path)


def main():
    folders = os.listdir(input_folder)
    aug_args = []

    for folder in folders:
        current_dir = os.path.join(input_folder, folder)
        aug_args.append([current_dir, folder])

    pool = ThreadPool(cpu)
    pool.map(glass_augmentation_wrapper, aug_args)
    pool.close()
    pool.join()


def glass_augmentation_wrapper(args):
    return glass_augmentation(*args)


def glass_augmentation(dir, folder):
    files = os.listdir(dir)
    mkdir_p(os.path.join(output_folder, folder))
    for file in files:
        full_path = os.path.join(dir, file)
        image = cv2.imread(full_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) == 0:
            continue
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        w, h, y, x = get_glass_with_and_start(shape, gray.shape[0])
        glasses = Image.open('filters/glasses.png').convert("RGBA")
        glasses = glasses.resize((w, h), Image.ANTIALIAS)
        glasses = glasses.filter(ImageFilter.SMOOTH_MORE)
        face = Image.fromarray(image[:, :, ::-1])
        glasses_offset = x, y
        face.paste(glasses, glasses_offset, glasses)
        face.save(os.path.join(output_folder, folder, glass_prefix + file))


def get_glass_with_and_start(shape, pic_width):
    width = shape[16][0] - shape[0][0]
    nose_mean = np.mean((shape[29][1])).astype(np.int)
    up_glass_mean = np.mean((shape[19][1])).astype(np.int)
    height = nose_mean - up_glass_mean
    center = pic_width // 2
    x = center - width // 2
    y = int(up_glass_mean)
    return width, height, y, x


if __name__ == '__main__':
    main()
