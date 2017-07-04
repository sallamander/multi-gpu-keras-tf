"""Test networks.py"""

from networks import VGG16

class TestVGG16():
    """Test VGG16 network"""

    def test_build(self):
        """Test build method"""

        # test across the different input shapes and number of classes from
        # cifar10, cifar100, and mnist
        datasets = {
            'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
            'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100},
            'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10}
        }

        for _, dataset_specs in datasets.items():
            vgg16 = VGG16()

            model = vgg16.build(
                input_shape=dataset_specs['input_shape'],
                num_classes=dataset_specs['num_classes']
            )
