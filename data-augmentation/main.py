import optparse
import os

import cv2
import dlib
from imutils import face_utils

parser = optparse.OptionParser()
parser.add_option("-f", "--folder")
parser.add_option("-p", "--percent", default=100)
(options, args) = parser.parse_args()
folder = options.folder

sep = os.path.sep

detector = dlib.get_frontal_face_detector()
predictor_data_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_data_path)


def main():
    files = os.listdir(folder)
    for file in files:
        full_path = folder + sep + file
        image = cv2.imread(full_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36], shape[39]
            right_eye = shape[42], shape[45]
            print(left_eye[0])
            print(left_eye[0])
            cv2.circle(image, tuple(left_eye[0]), 2, (0, 255, 0), -1)
            cv2.circle(image, tuple(left_eye[1]), 2, (0, 255, 0), -1)
            cv2.circle(image, tuple(right_eye[0]), 2, (0, 255, 0), -1)
            cv2.circle(image, tuple(right_eye[1]), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
