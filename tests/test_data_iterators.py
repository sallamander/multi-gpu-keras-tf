"""Test data_iterators.py"""

import numpy as np
from data_iterators import DataSetIterator

class TestDataSetIterator():
    """Test DataSetIterator"""

    nb_classes_dict = {'cifar10': 10, 'mnist': 10, 'cifar100': 100}
    dataset_names = ['cifar10', 'cifar100', 'mnist']

    def check_iter(self, iterator, batch_size, nb_total_batches):
        """Check the provided iterator

        Check:
            a. It returns a correctly sized batch
            b. It resets/re-shuffles appropriately

        :param iterator: an iterator that yields batches of data indefinitely
        :param batch_size: int for the size of the batches to be returned
        :param nb_total_batches: int for the total number of batches in the
         dataset; used to determine how many batches to pull (pull one more
         than this to check that the reset/re-shuffle works)
        """
 
        batches = []
        for _ in range(nb_total_batches + 1):
            x, y = next(iterator)

            assert x.shape[0] == batch_size
            assert y.shape[0] == batch_size
            batches.append((x, y))

        assert not np.array_equal(batches[0][0], batches[-1][0])
        assert not np.array_equal(batches[0][1], batches[-1][1])

    def test_load_data(self):
        """Test _load_data method"""

        for nb_obs in [8, 16, 32]:
            for name in self.dataset_names:
                dataset_iterator = DataSetIterator(name=name, nb_obs=nb_obs)

                img_size = (28, 28) if name == 'mnist' else (32, 32)
                expected_input_shape = (
                    (nb_obs, *img_size) if name == 'mnist' else
                    (nb_obs, *img_size, 3)
                )
                expected_output_shape = (nb_obs, self.nb_classes_dict[name])

                assert dataset_iterator.x_train.shape == expected_input_shape
                assert dataset_iterator.y_train.shape == expected_output_shape
                assert dataset_iterator.x_test.shape == expected_input_shape
                assert dataset_iterator.y_test.shape == expected_output_shape

    def test_get_train_iter(self):
        """Test get_train_iter method"""

        nb_total_batches = 2

        for batch_size in [32, 64, 128]:
            for name in self.dataset_names:
                dataset_iterator = DataSetIterator(
                    name=name, nb_obs=(batch_size * nb_total_batches)
                )

                train_iter = (
                    dataset_iterator.get_train_iter(batch_size=batch_size)
                )
                self.check_iter(train_iter, batch_size, nb_total_batches)

    def test_get_test_iter(self):
        """Test get_train_iter method"""

        nb_total_batches = 4

        for batch_size in [2, 4, 8]:
            for name in self.dataset_names:
                dataset_iterator = DataSetIterator(
                    name=name, nb_obs=(batch_size * nb_total_batches)
                )

                test_iter = (
                    dataset_iterator.get_test_iter(batch_size=batch_size)
                )
                self.check_iter(test_iter, batch_size, nb_total_batches - 1)
