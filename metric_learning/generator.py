import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow import keras
from tensorflow.python.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, all_files, all_labels, image_size, batch_size, classes_count, aug=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.all_files = all_files
        self.all_labels = all_labels
        self.classes_count = classes_count
        self.aug = aug

    def __len__(self):
        b_s = self.batch_size
        if self.aug:
            b_s = b_s // 2
        return len(self.all_files) // b_s

    def __getitem__(self, idx):
        b_s = self.batch_size
        if self.aug:
            b_s = b_s // 2
        batch_x = self.all_files[idx * b_s:(idx + 1) * b_s]
        batch_y = self.all_labels[idx * b_s:(idx + 1) * b_s]

        y_labels = keras.utils.to_categorical(np.array(batch_y), self.classes_count)
        images = np.array([
            np.array(resize(imread(file_name), (self.image_size, self.image_size))) / 255
            for file_name in batch_x])
        shape = np.array(batch_x).shape[0]
        if self.aug:
            y_flip_labels = keras.utils.to_categorical(np.array(batch_y), self.classes_count)
            fliped_images = np.array([
                np.flip(np.array(resize(imread(file_name), (self.image_size, self.image_size))) / 255, 1)
                for file_name in batch_x])
            images = np.concatenate((fliped_images, images))
            y_labels = np.concatenate((y_labels, y_flip_labels))
            shape *= 2

        dummy = np.zeros((shape, 1))

        return [images, y_labels], [y_labels, dummy]
