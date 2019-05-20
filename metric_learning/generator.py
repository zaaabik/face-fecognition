import numpy as np
from tensorflow import keras
from tensorflow.python.keras.utils import Sequence

from helpers.helpers import get_images


class Generator(Sequence):
    def __init__(self, all_files, all_labels, batch_size, classes_count, image_size=128):
        self.batch_size = batch_size
        self.image_size = image_size
        self.all_files = all_files
        self.all_labels = all_labels
        self.classes_count = classes_count

    def __len__(self):
        b_s = self.batch_size
        return len(self.all_files) // b_s

    def __getitem__(self, idx):
        b_s = self.batch_size
        batch_x = self.all_files[idx * b_s:(idx + 1) * b_s]
        batch_y = self.all_labels[idx * b_s:(idx + 1) * b_s]

        y_labels = keras.utils.to_categorical(np.array(batch_y), self.classes_count)
        images = get_images(batch_x, self.image_size)
        shape = np.array(batch_x).shape[0]

        dummy = np.zeros((shape, 1))

        return [images, y_labels], [y_labels, dummy]
