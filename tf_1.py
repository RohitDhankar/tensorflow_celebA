import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
#TensorFlow version: 2.2.0
# Set the random seed so things are reproducible
tf.random.set_seed(7) 

# Create 5 random input features
features = tf.random.normal((1, 5))

# Create random weights for our neural network
weights = tf.random.normal((1, 5))

# Create a random bias term for our neural network
bias = tf.random.normal((1, 1))

print('Features:\n', features)
print('\nWeights:\n', weights)
print('\nBias:\n', bias)
"""
Features:
 tf.Tensor([[ 0.5983449   0.06276207  0.14631742  0.48481876 -0.23572342]], shape=(1, 5), dtype=float32)

Weights:
 tf.Tensor([[-2.2733312  -1.6592104  -0.2633568  -0.80923414  1.0294315 ]], shape=(1, 5), dtype=float32)

Bias:
 tf.Tensor([[1.5749502]], shape=(1, 1), dtype=float32)
"""

"""

    features = tf.random.normal((1, 5)) creates a tensor with shape (1, 5), one row and five columns, that contains random values from a normal distribution with a mean of zero and standard deviation of one.

    weights = tf.random.normal((1, 5)) creates a tensor with shape (1, 5), one row and five columns, again containing random values from a normal distribution with a mean of zero and standard deviation of one.

    bias = tf.random.normal((1, 1)) creates a single random value from a normal distribution.

"""
# Check shape of the 3 Tensors 
print('Features Shape:', features.shape) #(1, 5)
print('Weights Shape:', weights.shape) #(1, 5)
print('Bias Shape:', bias.shape) #(1, 1)
#

def sigmoid_activation(param):
    """ Sigmoid activation function
        ---------
        param: tf.Tensor. Must be one of the following types: 
        bfloat16, half, float32, float64, complex64, complex128.
    """
    #result = 1/(1+tf.exp(-param))
    result = param * 10
    return result 
result = sigmoid_activation(22.11)
print(result)
#
