from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE
# tf.enable_eager_execution()


def preprocess_image(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image /= 255.0 - 0.5     # normalize to [-0.5,0.5] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_dataset():

    data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/'
    data_root = pathlib.Path(data_root)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    image_count = len(all_image_paths)

    sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/'
    sem_root = pathlib.Path(sem_root)

    all_vector_paths = list(sem_root.glob('*'))
    all_vector_paths = [str(path) for path in all_vector_paths]

    all_image_paths.sort()
    all_vector_paths.sort()

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    vectors = np.zeros((257, len(all_vector_paths)))
    for n, path in enumerate(all_vector_paths):
        v = np.loadtxt(path)
        vectors[:, n] = np.asarray(v)

    vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))

    image_vector_ds = tf.data.Dataset.zip((image_ds, vectors_ds))

    return image_vector_ds
