import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow import keras
from tensorflow.python.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, all_files, all_labels, image_size, batch_size, classes_count):
        self.batch_size = batch_size
        self.image_size = image_size
        self.all_files = all_files
        self.all_labels = all_labels
        self.classes_count = classes_count

    def __len__(self):
        return len(self.all_files) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.all_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.all_labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        y_labels = keras.utils.to_categorical(np.array(batch_y), self.classes_count)
        images = np.array([
            np.array(resize(imread(file_name), (self.image_size, self.image_size))) / 255
            for file_name in batch_x])
        y_flip_labels = keras.utils.to_categorical(np.array(batch_y), self.classes_count)
        fliped_images = np.array([
            np.flip(np.array(resize(imread(file_name), 0), (self.image_size, self.image_size))) / 255
            for file_name in batch_x])

        dummy = np.zeros((np.array(batch_x).shape[0] * 2, 1))
        images = np.concatenate(images, fliped_images)
        y_labels = np.concatenate(y_labels, y_flip_labels)

        return [images, y_labels], [y_labels, dummy]
