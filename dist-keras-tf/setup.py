"""Setup file"""

from setuptools import setup
from setuptools import find_packages

setup(
    name='dist-keras-tf',
    description='Distributed training with Keras using Tensorflow backend.',
    author='Sallamander',
    author_email='ssall@alumni.nd.edu',
    url='https://github.com/sallamander/dist-keras-tf',
    license='MIT',
    install_requires=['keras==2.0.2'],
    packages=find_packages()
)
