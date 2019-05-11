import optparse
import os

import cv2
import dlib

from helpers.helpers import mkdir_p

parser = optparse.OptionParser()
parser.add_option("-d", "--data")
parser.add_option("-o", "--out")
parser.add_option("-m", "--mode")
parser.add_option("-s", "--start_idx")
parser.add_option("-l", "--log")
(options, args) = parser.parse_args()
start_idx = int(options.start_idx or 0)
log = bool(options.log)


def main():
    if options.mode == 'crop':
        crop_faces_in_folder(options.data, options.out)


def crop_faces_in_folder(path, out_folder):
    folders = os.listdir(path)
    if not os.path.isdir(out_folder):
        mkdir_p(out_folder)
    folders_count = len(folders)
    for idx, folder in enumerate(folders):
        if idx < start_idx:
            continue
        print(f"{idx / folders_count * 100} %", flush=True, end='\r')
        files_path = path + os.path.sep + folder
        files = os.listdir(files_path)
        if not os.path.isdir(out_folder + os.path.sep + folder):
            os.mkdir(out_folder + os.path.sep + folder)
        for file in files:
            file_path = files_path + os.path.sep + file
            image = cv2.imread(file_path)
            if log:
                print(file_path)
            if image is None:
                continue
            face_detector = dlib.get_frontal_face_detector()
            faces = face_detector(image, 1)
            if len(faces) == 0:
                continue

            face = faces[0]
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            image = image[abs(y):y + h, abs(x):abs(x) + w]
            out_file_path = f'{out_folder}{os.path.sep}{folder}{os.path.sep}{file}'
            cv2.imwrite(out_file_path, image)


if __name__ == '__main__':
    main()
