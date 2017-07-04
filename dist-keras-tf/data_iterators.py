"""Dataset iterators used for training"""

import numpy as np
import keras.datasets
from keras.utils.np_utils import to_categorical


class DataSetIterator():
    """Wrapper around keras.datasets to make them generators"""

    def __init__(self, name, nb_obs=None):
        """Init

        :param name: str, one of 'cifar10', 'cifar100', or 'mnist'
        :param nb_obs: optional; int for the number of observations to retain
         from the training & testing sets; useful for testing, prototyping, or
         debugging; if None, retain the full training and testing sets
        """

        if name not in {'cifar10', 'cifar100', 'mnist'}:
            msg = ('Unsupported name option - must be one of \'cifar10\', '
                   '\'cifar100\', or \'mnist\'.')
            raise ValueError(msg)

        self.name = name
        self.x_train, self.y_train, self.x_test, self.y_test = (
            self._load_data(nb_obs=nb_obs)
        )

    def _load_data(self, nb_obs=None):
        """Load the dataset specified by self.name

        :param nb_obs: optional; int for the number of observations to retain
         from the training & testing sets; if None, retain the full training
         and testing sets
        :return: a tuple of 4 np.ndarrays (x_train, y_train, x_test, y_test)
        """

        dataset = getattr(keras.datasets, self.name)
        train_data, test_data = dataset.load_data()
        x_train, y_train = train_data[0] / 255., train_data[1]
        x_test, y_test = test_data[0] / 255., test_data[1]

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        if self.name == 'mnist':
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)

        if nb_obs:
            x_train = x_train[:nb_obs]
            y_train = y_train[:nb_obs]

            x_test = x_test[:nb_obs]
            y_test = y_test[:nb_obs]

        return x_train, y_train, x_test, y_test

    @property
    def input_shape(self):
        """Return the shape of one observation from x_train/x_test"""

        assert self.x_train[0].shape == self.x_test[0].shape
        return self.x_train[0].shape

    @property
    def output_shape(self):
        """Return the shape of one observation from y_train/y_test"""

        assert self.y_train[0].shape == self.y_test[0].shape
        return self.y_train[0].shape

    def get_train_iter(self, batch_size=32):
        """Return a generator that yields batches of training data indefinitely

        :param batch_size: int for the number of training examples to return
         per iteration
        :return: tuple of (np.ndarray, np.ndarray) pulled from self.x_train,
         self.y_train
        """

        nb_train_samples = len(self.x_train)
        idx_min = 0

        while True:
            idx_max = idx_min + batch_size

            if idx_max > nb_train_samples:
                idx_min = 0
                idx_max = idx_min + batch_size

                idx_shuffle = (
                    np.random.RandomState(529).permutation(nb_train_samples)
                )

                self.x_train = self.x_train[idx_shuffle]
                self.y_train = self.y_train[idx_shuffle]

            x_train = self.x_train[idx_min:idx_max]
            y_train = self.y_train[idx_min:idx_max]
            idx_min += batch_size

            yield x_train, y_train

    def get_test_iter(self, batch_size=32):
        """Return a generator that yields batches of test data indefinitely

        :param batch_size: int for the number of training examples to return
         per iteration
        :return: tuple of (np.ndarray, np.ndarray) pulled from self.x_test,
         self.y_test
        """

        idx_min = 0

        while True:
            idx_max = idx_min + batch_size

            if idx_max > len(self.x_test):
                idx_min = 0
                idx_max = idx_min + batch_size

            x_test = self.x_test[idx_min:idx_max]
            y_test = self.y_test[idx_min:idx_max]
            idx_min += batch_size

            yield x_test, y_test
