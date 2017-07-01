"""Keras MultiGPUModel"""

import tensorflow as tf
from keras.layers import Concatenate, Lambda
from keras.models import Model


class MultiGPUModel(Model):
    """Class that builds/trains a model on multiple GPUs

    Inspired by the keras_exp/multigpu/_multigpu.py.ModelMGPU class from:
        https://github.com/avolkov1/keras_experiments/
    """

    def __init__(self, serial_model, gpu_ids, batch_size):
        """Init

        :param serial_model: Keras Model instance that will be parallelized
         across the given gpu_ids
        :param gpu_ids: list of integers to parallelize the serial_model across
        :param batch_size: integer for the batch size that will be run on each
         GPU
        """

        self.serial_model = serial_model
        self._parallelize_model(serial_model, gpu_ids, batch_size)

    def __getattribute__(self, key):
        """Return the key attribute from self.[serial/parallelized]_model

        :param key: attribute/method to access
        """

        # attributes where we really want to access the underlying serial_model
        serial_attributes = {
            'load_weights', 'save_weights',
            'summary', 'to_yaml', 'save', 'to_json',
        }

        if key in serial_attributes:
            return getattr(self.serial_model, key)

        return super().__getattribute__(key)

    def _parallelize_model(self, model, gpu_ids, batch_size):
        """Parallelize the model over the given gpu_ids

        Note: This is largely copied from:

        https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

        :param model: Keras Model instance
        :param gpu_ids: list of integers to run the model on
        :param batch_size: int holding the batch size to use during training;
         if multiple gpus are passed in, one batch with batch_size will be run
         on each gpu

        TODO: Put what is returned here
        """

        all_sliced_outputs = []
        for gpu_id in gpu_ids:
            with tf.device('/gpu:{}'.format(gpu_id)):

                sliced_inputs = []
                # ignore the defining of the cell (idx_min, idx_max) variable;
                # this should be fine since we call the Lambda in the next line
                # pylint: disable=cell-var-from-loop
                for model_input in model.inputs:
                    idx_min = gpu_id * batch_size
                    idx_max = (gpu_id + 1) * batch_size
                    input_slice = Lambda(
                        lambda x: x[idx_min:idx_max],
                        lambda shape: shape
                    )(model_input)
                    sliced_inputs.append(input_slice)

                sliced_outputs = model(sliced_inputs)
                all_sliced_outputs.append(sliced_outputs)

        with tf.device('/cpu:0'):
            outputs = Concatenate(axis=0)(all_sliced_outputs)

            super().__init__(inputs=model.inputs, outputs=outputs)
