import optparse
import os

import cv2
import dlib
from imutils.face_utils import FaceAligner

parser = optparse.OptionParser()
parser.add_option('--path')
parser.add_option('--out')
(options, args) = parser.parse_args()
path = options.path
out = options.out


def main():
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(predictor=predictor, desiredFaceHeight=128, desiredFaceWidth=128)
    detector = dlib.get_frontal_face_detector()
    #
    folders = os.listdir(path)

    if not os.path.isdir(out):
        os.mkdir(out)
    for idx, folder in enumerate(folders):
        print(idx)
        files_path = path + os.path.sep + folder
        files = os.listdir(files_path)
        res_folder = out + os.path.sep + folder
        if not os.path.isdir(res_folder):
            os.mkdir(res_folder)
        for file in files:
            file_path = files_path + os.path.sep + file
            img = cv2.imread(file_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img_gray, 0)
            if len(rects) == 1:
                aligned_face = face_aligner.align(img, img_gray, rects[0])
                cv2.imwrite(res_folder + os.path.sep + file, aligned_face)


if __name__ == '__main__':
    main()
