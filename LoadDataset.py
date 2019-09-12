from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from FaceNet3D import FaceNet3D as Helpers
import pathlib
import numpy as np
import random


class LoadDataset(Helpers):
    tf.compat.v1.enable_eager_execution()

    def __init__(self):
        super().__init__()
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def preprocess_image(self, image, test_dataset):
        """
        Decodes tensor and cast to tf.float32
        Maps color channels to [0,5, 0,5]

        :param image: Tensor("ReadFile:0", shape=(), dtype=string)
        :param test_dataset: Boolean, if creating test dataset, reshape tensor
        :return: Tensor("truediv:0", dtype=float32)
        """
        image = tf.image.decode_image(image, channels=self.COLOR_CHANNELS, dtype=tf.dtypes.float32)
        image = image/255.0 - 0.5
        image = tf.reshape(image, shape=[self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS])

        if test_dataset:
            image = tf.reshape(image, shape=[1, self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS])

        return image

    def load_and_preprocess_image(self, path):
        """
        Reads string path into image string and calls preprocess function to cast into image tensor

        :param path: Tensor("args_0:0", shape=(), dtype=string)
        :return: Tensor("truediv:0", dtype=float32)
        """
        image = tf.io.read_file(path)
        return self.preprocess_image(image, False)

    def load_and_preprocess_image_4d(self, path):
        """
        Reads string path into image string and calls preprocess function to cast into image tensor
        Used for model evaluation.

        :param path: Tensor("args_0:0", shape=(), dtype=string)
        :return: Tensor("truediv:0", dtype=float32)
        """
        image = tf.io.read_file(path)
        return self.preprocess_image(image, True)

    def load_dataset_batches(self):
        """
        Read images and vectors (from txt files) and zips them together in a Tensorflow Dataset
        Images and vectors should be in different directories

        :return: tf.data.Dataset with pairs (Image, Semantic Code Vector)
        """

        if self._case == 'training':
            data_root = self.data_root + 'training/'
            sem_root = self.sem_root + 'training/'
        elif self._case == 'bootstrapping':
            data_root = self.bootstrapping_path + 'images/'
            sem_root = self.bootstrapping_path + 'semantic/'
        elif self._case == 'validation':
            data_root = self.data_root + 'validation/'
            sem_root = self.sem_root + 'validation/'
        else:
            data_root = self.data_root
            sem_root = self.sem_root

        data_root = pathlib.Path(data_root)

        all_image_paths = list(data_root.glob('*'))
        all_image_paths = [str(path) for path in all_image_paths]

        sem_root = pathlib.Path(sem_root)

        all_vector_paths = list(sem_root.glob('*'))
        all_vector_paths = [str(path) for path in all_vector_paths]

        all_image_paths.sort()
        all_vector_paths.sort()

        all_image_paths = all_image_paths[20000:30000]
        all_vector_paths = all_vector_paths[20000:30000]

        image_count = len(all_image_paths)
        print("Dataset containing %d pairs of Images and Vectors." % image_count)
        vector_count = len(all_vector_paths)
        # print("Dataset containing %d pairs of Images and Vectors." % vector_count)
        if image_count != vector_count:
            print("Dataset not compatible")
            quit()

        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=64)

        vectors = np.zeros((self.scv_length, len(all_vector_paths)))
        for n, path in enumerate(all_vector_paths):
            v = np.loadtxt(path)
            vectors[:, n] = v

        vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))
        image_vector_ds = tf.data.Dataset.zip((image_ds, vectors_ds))

        return image_vector_ds

    def load_dataset_single_image(self, _case):
        """
        Read images and vectors (from txt files) and zips them together in a Tensorflow Dataset
        Images and vectors should be in different directories
        Used for model evaluation.

        :return: tf.data.Dataset with pairs (Image, Semantic Code Vector)
        """

        if self._case == 'training':
            data_root = self.data_root + 'training/'
            sem_root = self.sem_root + 'training/'
        elif self._case == 'bootstrapping':
            data_root = self.bootstrapping_path + 'images/'
            sem_root = self.bootstrapping_path + 'semantic/'
        elif self._case == 'validation':
            data_root = self.data_root + 'validation/'
            sem_root = self.sem_root + 'validation/'
        else:
            data_root = self.data_root
            sem_root = self.sem_root

        data_root = pathlib.Path(data_root)

        all_image_paths = list(data_root.glob('*'))
        all_image_paths = [str(path) for path in all_image_paths]

        sem_root = pathlib.Path(sem_root)

        all_vector_paths = list(sem_root.glob('*'))
        all_vector_paths = [str(path) for path in all_vector_paths]

        all_image_paths.sort()
        all_vector_paths.sort()

        image_count = len(all_image_paths)
        print("Dataset containing %d pairs of Images and Vectors." % image_count)
        vector_count = len(all_vector_paths)
        # print("Dataset containing %d pairs of Images and Vectors." % vector_count)
        if image_count != vector_count:
            print("Dataset not compatible")
            quit()

        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image_4d, num_parallel_calls=self.AUTOTUNE)

        vectors = np.zeros((self.scv_length, len(all_vector_paths)))
        for n, path in enumerate(all_vector_paths):
            v = np.loadtxt(path)
            vectors[:, n] = v

        vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))
        image_vector_ds = tf.data.Dataset.zip((image_ds, vectors_ds))

        return image_vector_ds

    def load_data_for_expression(self):
        data_root = pathlib.Path("./DATASET/expression/")
        all_vector_paths = list(data_root.glob('*/eb3*.txt'))
        all_vector_paths = [str(path) for path in all_vector_paths]
        random.shuffle(all_vector_paths)

        vector_count = len(all_vector_paths)
        print("Dataset containing %d pairs of Vectors and Labels." % vector_count)

        data_root = pathlib.Path("./DATASET/expression/")
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_vectors_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_vector_paths]

        # print("First 10 labels indices: ", all_vectors_labels[:10])
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_vectors_labels, tf.int64))

        vectors = np.zeros((self.expression_dim, len(all_vector_paths)))
        for n, path in enumerate(all_vector_paths):
            v = np.loadtxt(path)
            vectors[:, n] = v

        vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))

        vectors_labels_ds = tf.data.Dataset.zip((vectors_ds, label_ds))

        return vectors_labels_ds

    def load_data_for_expression_evaluate(self):
        data_root = pathlib.Path("./DATASET/expression/")
        all_vector_paths = list(data_root.glob('*/eb1*.txt'))
        all_vector_paths = [str(path) for path in all_vector_paths]
        random.shuffle(all_vector_paths)

        vector_count = len(all_vector_paths)
        print("Dataset containing %d pairs of Vectors and Labels." % vector_count)

        data_root = pathlib.Path("./DATASET/expression/")
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_vectors_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_vector_paths]

        # print("First 10 labels indices: ", all_vectors_labels[:10])
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_vectors_labels, tf.int64))

        vectors = np.zeros((self.expression_dim, len(all_vector_paths)))
        # vector = tf.reshape(vector, shape=[1, self.expression_dim])
        for n, path in enumerate(all_vector_paths):
            v = np.loadtxt(path)
            vectors[:, n] = v

        vectors_ds = tf.data.Dataset.from_tensor_slices(np.transpose(vectors))

        vectors_labels_ds = tf.data.Dataset.zip((vectors_ds, label_ds))

        return vectors_labels_ds
