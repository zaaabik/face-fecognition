import errno
import optparse
import os

import cv2
import dlib
from imutils.face_utils import FaceAligner

from helpers.helpers import mkdir_p

parser = optparse.OptionParser()
parser.add_option('--path')
parser.add_option('--out')
parser.add_option('--skip', type='int', default=0)
(options, args) = parser.parse_args()
path = options.path
out = options.out
skip = options.skip or 0


def main():
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(predictor=predictor, desiredLeftEye=(0.315, 0.315), desiredFaceWidth=128)
    detector = dlib.get_frontal_face_detector()

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
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(img_gray, 0)
                if len(rects) == 1:
                    aligned_face = face_aligner.align(img, img_gray, rects[0])
                    cv2.imwrite(res_folder + os.path.sep + file, aligned_face)
            except Exception as e:
                print(file_path)
                print(str(e))


if __name__ == '__main__':
    main()
