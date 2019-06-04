import optparse
import os
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import dlib
import numpy as np
import pandas
from PIL import Image
from PIL import ImageFilter
from imutils import face_utils

from helpers.helpers import mkdir_p

meta_path = 'meta/meta.csv'

parser = optparse.OptionParser()
parser.add_option("-i", "--input")
parser.add_option("-o", "--output", type='string')
parser.add_option("-b", "--beard", default=False)
parser.add_option("-c", "--cpu", type='int', default=4)
(options, args) = parser.parse_args()
input_folder = options.input
output_folder = options.output
beard = options.beard
cpu = options.cpu

sep = os.path.sep

glass_prefix = '_glass_'

detector = dlib.get_frontal_face_detector()
predictor_data_path = '../shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_data_path)
glass_max_count = 4
beard_max_count = 3


def main():
    folders = os.listdir(input_folder)
    meta = parse_gender_info()

    aug_args = []
    for folder in folders:
        integer_folder = int(folder[1:])
        print(folder)
        wh = np.where(meta[:, 0] == integer_folder)
        is_male = meta[wh[0][0], 1]
        current_dir = os.path.join(input_folder, folder)
        aug_args.append([current_dir, folder, is_male])

    pool = ThreadPool(cpu)
    pool.map(glass_augmentation_wrapper, aug_args)
    pool.close()
    pool.join()


def glass_augmentation_wrapper(args):
    return glass_augmentation(*args)


def glass_augmentation(dir, folder, is_male):
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
        face = Image.fromarray(image[:, :, ::-1])

        if is_male and beard:
            w, h, y, x = get_beard_size_and_start(shape, gray.shape[0])
            random_number = np.random.randint(0, beard_max_count)
            beard = Image.open(f'filters/beard{random_number}.png').convert("RGBA")
            beard = beard.resize((w, h), Image.ANTIALIAS)
            beard_offset = x, y
            face.paste(beard, beard_offset, beard)

        w, h, y, x = get_glass_size_and_start(shape, gray.shape[0])
        glasses_offset = x, y
        random_number = np.random.randint(0, glass_max_count)
        glasses = Image.open(f'filters/glasses{random_number}.png').convert("RGBA")
        glasses = glasses.resize((w, h), Image.ANTIALIAS)
        glasses = glasses.filter(ImageFilter.SMOOTH_MORE)
        face.paste(glasses, glasses_offset, glasses)

        face.save(os.path.join(output_folder, folder, glass_prefix + file))


def parse_gender_info():
    meta = pandas.read_csv(meta_path, quotechar=r'"', skipinitialspace=True)
    meta['Class_ID'] = meta['Class_ID'].str.replace("n", '')
    meta['Class_ID'] = meta['Class_ID'].astype(np.int)
    meta['Gender'] = meta['Gender'].replace({'m': True, 'f': False})
    meta = meta[['Class_ID', 'Gender']]
    meta = meta.values
    return meta


# (x,y) start left top
def get_glass_size_and_start(shape, pic_width):
    width = shape[16][0] - shape[0][0]
    nose_mean = np.mean((shape[29][1])).astype(np.int)
    up_glass_mean = np.mean((shape[19][1])).astype(np.int)
    height = nose_mean - up_glass_mean
    center = pic_width // 2
    x = center - width // 2
    y = int(up_glass_mean)
    return width, height, y, x


def get_beard_size_and_start(shape, pic_width):
    height = shape[8][1] - shape[0][1]
    width = shape[16][0] - shape[0][0]
    x = shape[0][0]
    y = shape[0][1]
    return width, height, y, x


if __name__ == '__main__':
    main()
