import keras
import numpy as np
from tensorflow.python.keras.callbacks import Callback

from validators.lfw import read_pairs_file, create_pairs, read_images


class ValidateOnLfw(Callback):
    def __init__(self, pairs_path, lfw_path, classes_count):
        self.pairs = pairs_path
        self.lfw_path = lfw_path
        self.classes_count = classes_count

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0 or True:
            pairs = read_pairs_file(self.pairs)
            pairs, positive = create_pairs(self.lfw_path, pairs)
            resnset = self.model
            count = len(pairs)
            pairs = np.array(pairs)
            first_images = pairs[:, 0]
            second_images = pairs[:, 1]
            first_images = read_images(first_images)
            second_images = read_images(second_images)
            dummy = np.zeros(len(first_images))
            dummy = keras.utils.to_categorical(dummy)
            first_inferences = resnset.predict(first_images, dummy)
            second_inferences = resnset.predict(second_images, dummy)

            first_images_flipped = np.flip(first_images, 2)
            second_images_flipped = np.flip(second_images, 2)
            first_inferences_flipped = resnset.predict(first_images_flipped)
            second_inferences_flipped = resnset.predict(second_images_flipped)
            first_inferences = (first_inferences + first_inferences_flipped) / 2
            second_inferences = (second_inferences + second_inferences_flipped) / 2

            distanses = np.linalg.norm(first_inferences - second_inferences, axis=1).flatten()
            positive = np.array(positive).flatten()

            thresholds = np.array(np.arange(0, 3, 0.01))
            thr = np.zeros((len(thresholds), len(positive)), dtype=float)
            for idx, val in enumerate(thresholds):
                thr[idx, :] = val

            res = (thr - distanses)
            res = np.where(res > 0, True, False)
            thrs_acc = []
            for i in range(0, res.shape[0]):
                right_answers = (res[i] == positive).sum()
                accuracy = right_answers / count
                thrs_acc.append(accuracy)
            thrs_acc = np.array(thrs_acc)

            best_thr_arg = np.argmax(thrs_acc)
            print('best thr ', thresholds[best_thr_arg])
            print('best accuracy', np.max(thrs_acc))
