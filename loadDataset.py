from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pathlib
import numpy as np


def preprocess_image(image, test_dataset):
    """
    Decodes tensor and cast to tf.float32
    Maps color channels to [-0.5, 0.5]

    :param image: Tensor("ReadFile:0", shape=(), dtype=string)
    :param test_dataset: Boolean, if creating test dataset, reshape tensor
    :return: Tensor("truediv:0", dtype=float32)
    """
    image = tf.image.decode_image(image, channels=3, dtype=tf.dtypes.float32)
    # print(image.shape)
    # image = tf.cast(image, dtype=tf.float32)
    image = image/255.0     # normalize to [-0.5,0.5] range

    if test_dataset:
        image = tf.reshape(image, shape=[1, 240, 240, 3])

    return image


def load_and_preprocess_image(path):
    """
    Reads string path into image string and calls preprocess function to cast into image tensor

    :param path: Tensor("args_0:0", shape=(), dtype=string)
    :return: Tensor("truediv:0", dtype=float32)
    """
    image = tf.io.read_file(path)
    return preprocess_image(image, False)


def load_and_preprocess_image_4d(path):
    """
    Reads string path into image string and calls preprocess function to cast into image tensor

    :param path: Tensor("args_0:0", shape=(), dtype=string)
    :return: Tensor("truediv:0", dtype=float32)
    """
    image = tf.io.read_file(path)
    return preprocess_image(image, True)


def load_dataset_batches(_case):
    """
    Read images and vectors (from txt files) and zips them together in a Tensorflow Dataset
    Images and vectors should be in different directories

    :return: tf.data.Dataset with pairs (Image, Semantic Code Vector)
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    if _case == 'training':
        data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/training/'
        sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/training/'
    elif _case == 'bootstrapping':
        data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/bootstrapping'
        sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/bootstrapping'
    elif _case == 'validation':
        data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/validation'
        sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/validation'
    else:
        data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/'
        sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/'

    data_root = pathlib.Path(data_root)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    image_count = len(all_image_paths)
    print("Dataset containing %d pairs of Images and Vectors." % image_count)

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
        vectors[:, n] = v

    vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))
    image_vector_ds = tf.data.Dataset.zip((image_ds, vectors_ds))

    return image_vector_ds


def load_dataset_single_image():
    """
    Read images and vectors (from txt files) and zips them together in a Tensorflow Dataset
    Images and vectors should be in different directories

    :return: tf.data.Dataset with pairs (Image, Semantic Code Vector)
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/'
    data_root = pathlib.Path(data_root)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    image_count = len(all_image_paths)
    print("Dataset containg %d pairs of Images and Vectors." % image_count)

    sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/'
    sem_root = pathlib.Path(sem_root)

    all_vector_paths = list(sem_root.glob('*'))
    all_vector_paths = [str(path) for path in all_vector_paths]

    all_image_paths.sort()
    all_vector_paths.sort()

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image_4d, num_parallel_calls=AUTOTUNE)

    vectors = np.zeros((257, len(all_vector_paths)))
    for n, path in enumerate(all_vector_paths):
        v = np.loadtxt(path)
        vectors[:, n] = np.asarray(v)

    vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))

    image_vector_ds = tf.data.Dataset.zip((image_ds, vectors_ds))

    return image_vector_ds
