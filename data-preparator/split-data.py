import argparse
import os

import cv2
import dlib

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data")
parser.add_argument("-l", "--labels")
args = parser.parse_args()


def main():
    # crop_faces_in_folder(r'D:\dataset\vgg\my_photos')
    more_then()


def more_then():
    path = r'D:\dataset\vgg\test\\'
    result_path = r'D:\dataset\vgg\prepared\\'
    folders = os.listdir(path)
    hog_face_detector = dlib.get_frontal_face_detector()
    for i in folders:
        l = os.listdir(path + i)
        result_folder = result_path + i + '\\'
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        for img in l:
            image_path = path + i + '\\' + img
            image = cv2.imread(image_path)
            faces_hog = hog_face_detector(image, 1)

            if (len(faces_hog)) == 0:
                continue

            face = faces_hog[0]
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            image = image[abs(y):y + h, abs(x):abs(x) + w]
            image = cv2.resize(image, (50, 50))
            output_path = result_path + i + '\\' + img
            cv2.imwrite(output_path, image)


def crop_faces_in_folder(path):
    files = os.listdir(path)
    out_folder = os.path.dirname(path) + '\out'
    os.mkdir(out_folder)

    for file in files:
        image = cv2.imread(path + '/' + file)
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(image, 1)
        if len(faces) == 0:
            return

        face = faces[0]
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        image = image[abs(y):y + h, abs(x):abs(x) + w]
        output_image = cv2.resize(image, (50, 50))
        cv2.imwrite(out_folder + '\\' + file, output_image)


if __name__ == '__main__':
    main()
