from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import random
import sys
import cv2
import numpy as np
tf.enable_eager_execution()
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
    image = tf.io.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image/255.0 - 0.5   # normalize between -0.5 and 0.5

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_vector(path):
    # print(path)
    vector = tf.cast(tf.read_file(path), dtype=tf.float32)
    return vector


data_root = '/home/anapt/PycharmProjects/thesis/DATASET/'
data_root = pathlib.Path(data_root)

sorted_images = sorted(item.name for item in data_root.glob('images/*'))
sorted_vectors = sorted(item.name for item in data_root.glob('semantic/*'))
all_image_sorted = [str(path) for path in sorted_images]
all_vector_sorted = [str(path) for path in sorted_vectors]


all_image_paths = [pathlib.Path.joinpath(pathlib.Path(data_root, 'images/', path)) for path in all_image_sorted]
all_vector_paths = [pathlib.Path.joinpath(pathlib.Path(data_root, 'semantic/', path)) for path in all_vector_sorted]
print(type(all_image_paths))
all_image_paths = np.asarray(all_image_paths, dtype=np.unicode)
all_vector_paths = np.asarray(all_vector_paths, dtype=np.unicode)
# print(all_image_paths)


# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = path_ds.map(load_and_preprocess_image)
# # print(image_ds)
#
# # vector = load_vector(all_vector_paths)
# path_2ds = tf.data.Dataset.from_tensor_slices(all_vector_paths)
# vector_ds = path_2ds.map(load_vector)
# print(vector_ds)
# print(vector_ds.take(1))
# print(vector_ds)
# vector = tf.data.TextLineDataset(all_vector_paths[0])

# vec = tf.io.decode_csv(vec, record_defaults='float32', field_delim=',')

# tf.io.decode_csv(vec, record_defaults=tf.float32, field_delim=',', use_quote_delim=True)
# print(vec)
# for i in range(0, len(all_vector_paths)):
#     vector = tf.cast(np.loadtxt(all_vector_paths[i]), dtype=tf.float32)

# image_vector_ds = tf.data.Dataset.zip((image_ds, vector_ds))
# print(image_vector_ds)

# plt.figure(figsize=(8,8))
# for n, im in enumerate(image_ds.take(4)):
#     plt.subplot(2,2,n+1)
#     plt.imshow(tf.cast(im, dtype=tf.float32))
#     # plt.imshow(im.reshape(im.shape[0], im.shape[1]), cmap=plt.cm.Greys)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
# for n, im in enumerate(image_ds.take(4)):
#     tf.print(im, output_stream=sys.stderr)
#
# for n, im in enumerate(vector_ds.take(4)):
#     tf.print(im, output_stream=sys.stderr)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_vector_paths))


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), load_vector(label)


image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

# plt.figure(figsize=(8,8))
# for n, im in enumerate(image_label_ds.take(4)):
#     plt.subplot(2,2,n+1)
#     plt.imshow(tf.cast(im, dtype=tf.float32))
#     # plt.imshow(im.reshape(im.shape[0], im.shape[1]), cmap=plt.cm.Greys)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
for n, im in enumerate(image_label_ds.take(4)):
    tf.print(im, output_stream=sys.stderr)
#
# for n, im in enumerate(vector_ds.take(4)):
#     tf.print(im, output_stream=sys.stderr)