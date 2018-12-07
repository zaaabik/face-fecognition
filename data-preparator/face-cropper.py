import optparse
import os

import cv2
import dlib

parser = optparse.OptionParser()
parser.add_option("-d", "--data")
parser.add_option("-o", "--out")
parser.add_option("-m", "--mode")
(options, args) = parser.parse_args()


def main():
    if options.mode == 'crop':
        crop_faces_in_folder(options.data, options.out)


def crop_faces_in_folder(path, out_folder):
    folders = os.listdir(path)
    out_folder = os.path.dirname(path) + os.path.sep + out_folder
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    folders_count = len(folders)
    for idx, folder in enumerate(folders):
        print(f"{idx / folders_count * 100} %", flush=True, end='\r')
        files_path = path + os.path.sep + folder
        files = os.listdir(files_path)
        if not os.path.isdir(out_folder + os.path.sep + folder):
            os.mkdir(out_folder + os.path.sep + folder)
        for file in files:
            file_path = files_path + os.path.sep + file
            image = cv2.imread(file_path)
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
