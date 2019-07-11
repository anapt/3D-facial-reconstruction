from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import random
import cv2
tf.enable_eager_execution()


data_root = '/home/anapt/PycharmProjects/thesis/DATASET/'
data_root = pathlib.Path(data_root)
# print(data_root)
# all_image_paths = list(data_root.glob('images/*'))
# print(all_image_paths)
# all_vector_paths = list(data_root.glob('semantic/*'))
# all_image_paths = [str(path) for path in all_image_paths]
# all_vector_paths = [str(path) for path in all_vector_paths]

sorted_images = sorted(item.name for item in data_root.glob('images/*'))
sorted_vectors = sorted(item.name for item in data_root.glob('semantic/*'))
all_image_sorted = [str(path) for path in sorted_images]
all_vector_sorted = [str(path) for path in sorted_vectors]

print(all_image_sorted[0:10])
print(all_vector_sorted[0:10])
image_count = len(all_image_sorted)
print(image_count)

img_path = all_image_sorted[0]
img_path = pathlib.Path.joinpath(data_root, 'images/', img_path)
print(img_path)
img_raw = tf.read_file(str(img_path))
img_tensor = tf.io.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)
