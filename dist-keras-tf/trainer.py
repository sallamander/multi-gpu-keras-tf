"""Model training class"""

import os
import tensorflow as tf
from keras.models import Model


class KerasTrainer():
    """Trainer for Keras models"""

    def __init__(self, output_dir):
        """Init

        The point of this class is to house functionality to possibly build out
        trainers later. It might be a little unncessary for this to be in a
        class right now, but ¯\_(ツ)_/¯.

        :param output_dir: str holding the directory path to save model output
         to
        """

        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def _set_cuda_devices(gpu_id):
        """Set CUDA related environment variables before running models

        :param gpu_id: int holding a GPU ID
        """

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    def _save_model(self, model):
        """Save the model to a YML and weights file

        :param model: Keras Model object
        """

        fpath_model_yml = os.path.join(self.output_dir, 'model.yml')
        fpath_weights = os.path.join(self.output_dir, 'weights.h5')

        yaml_str = model.to_yaml()
        with open(fpath_model_yml, 'w') as f:
            f.write(yaml_str)

        model.save_weights(fpath_weights)

    def train(self, network, dataset_iterator, compile_args, fit_args, 
              gpu_id):
        """Train a model specified by network using data from dataset_iterator


        :param network: object with a build() method that returns the inputs &
         outputs for a Keras model
        :param dataset_iterator: DataSetIterator instance
        :param compile_args: dictionary of key/value pairs to pass as keyword
         arguments to the compile method of a Keras model
        :param fit_args: dictionary of key/value pairs to pass as keyword
         arguments to the fit method of a Keras model
        :param gpu_id: integer for the GPU to train the model on
        """

        input_shape = dataset_iterator.input_shape
        num_classes = dataset_iterator.output_shape[-1]

        self._set_cuda_devices(gpu_id)
        with tf.device('/gpu:{}'.format(gpu_id)):
            model = network.build(input_shape, num_classes)
        model.compile(**compile_args)

        batch_size=fit_args.pop('batch_size')
        train_iter = dataset_iterator.get_train_iter(batch_size)
        steps_per_epoch = max(
            dataset_iterator.x_train.shape[0] // batch_size, 1
        )

        test_iter = dataset_iterator.get_test_iter(batch_size)
        validation_steps = max(
            dataset_iterator.x_test.shape[0] // batch_size, 1
        )

        model.fit_generator(
            generator=train_iter,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_iter,
            validation_steps=validation_steps,
            **fit_args
        )

        self._save_model(model)
