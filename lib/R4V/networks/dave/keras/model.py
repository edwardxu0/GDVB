"""
The DAVE-2 driving model.

This file is modified from DeepXplore to include only the original dave model.
DeepXplore: https://github.com/peikexin9/deepxplore
"""
import os
import tensorflow as tf

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, Multiply
from keras.models import Model


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def DAVE(input_tensor=None, load_weights=True):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(
        24,
        (5, 5),
        padding="valid",
        activation="relu",
        strides=(2, 2),
        name="block1_conv1",
    )(input_tensor)
    x = Convolution2D(
        36,
        (5, 5),
        padding="valid",
        activation="relu",
        strides=(2, 2),
        name="block1_conv2",
    )(x)
    x = Convolution2D(
        48,
        (5, 5),
        padding="valid",
        activation="relu",
        strides=(2, 2),
        name="block1_conv3",
    )(x)
    x = Convolution2D(
        64,
        (3, 3),
        padding="valid",
        activation="relu",
        strides=(1, 1),
        name="block1_conv4",
    )(x)
    x = Convolution2D(
        64,
        (3, 3),
        padding="valid",
        activation="relu",
        strides=(1, 1),
        name="block1_conv5",
    )(x)
    x = Flatten(name="flatten")(x)
    x = Dense(1164, activation="relu", name="fc1")(x)
    x = Dense(100, activation="relu", name="fc2")(x)
    x = Dense(50, activation="relu", name="fc3")(x)
    x = Dense(10, activation="relu", name="fc4")(x)
    x = Dense(1, name="before_prediction")(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    m = Model(input_tensor, x)
    if load_weights:
        weights_path = os.path.join(os.path.dirname(__file__), "./Model1.h5")
        m.load_weights(weights_path)

    return m
