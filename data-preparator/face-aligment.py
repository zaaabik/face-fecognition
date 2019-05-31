import json
import optparse
import os
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import dlib

from helpers.facealigner import FaceAligner
from helpers.helpers import mkdir_p, parse_landmarks

parser = optparse.OptionParser()
parser.add_option('--path')
parser.add_option('--out')
parser.add_option('--cpu', type='int', default=4)
parser.add_option('--file', type='string')
parser.add_option('--skip', type='int', default=0)
(options, args) = parser.parse_args()
path = options.path
out = options.out
cpu = options.cpu
file = options.file
skip = options.skip or 0

predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(predictor=predictor, desiredLeftEye=(0.315, 0.315), desiredFaceWidth=128)
detector = dlib.get_frontal_face_detector()


def align_folder():
    folders = os.listdir(path)

    if not os.path.isdir(out):
        mkdir_p(out)

    for idx, folder in enumerate(folders):
        if idx < skip:
            continue
        print(idx)
        files_path = path + os.path.sep + folder
        files = os.listdir(files_path)
        res_folder = out + os.path.sep + folder
        if not os.path.isdir(res_folder):
            os.mkdir(res_folder)
        for file in files:
            file_path = files_path + os.path.sep + file
            try:
                img = cv2.imread(file_path)
                parse_landmarks()
                aligned_face = face_aligner.align(img, img_gray, None, )
            except Exception as e:
                print(file_path)
                print(str(e))


def align_from_file():
    with open(file) as images_file:
        feature_file = json.load(images_file)
        paths = feature_file['path']

    pool = ThreadPool(cpu)
    pool.map(align_image, paths)
    pool.close()
    pool.join()


def align_image(file_path):
    output_directory = os.path.dirname(file_path)
    output_directory = os.path.join(out, output_directory)
    output_file = os.path.join(out, file_path)
    mkdir_p(output_directory)
    try:
        full_image_path = os.path.join(path, file_path)
        img = cv2.imread(full_image_path)
        file_landmark_path = full_image_path + '.json'

        aligned_face = img
        if os.path.isfile(file_landmark_path):
            try:
                rights_eye, left_eye = parse_landmarks(file_landmark_path)
                aligned_face = face_aligner.align(img, right=rights_eye, left=left_eye)
            except AttributeError as e:
                print(file_path)
                print(e)

        cv2.imwrite(output_file, aligned_face)
    except Exception as e:
        print(file_path)
        print(str(e))


if __name__ == '__main__':
    if file is not None:
        align_from_file()
    else:
        align_folder()
