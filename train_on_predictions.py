from FaceNet3D import FaceNet3D as Helpers
import matplotlib.pyplot as plt
from LoadDataset import LoadDataset
import tensorflow as tf
import os
import numpy as np
import pathlib

emotions = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

for emotion in emotions:
    data_root = './DATASET/images/training/{}/'.format(emotion)
    data_root = pathlib.Path(data_root)

    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    print(all_image_paths)

    for i, path in enumerate(all_image_paths):
        vector = np.loadtxt("./DATASET/semantic/training/x_" + path[-10:-4] + ".txt")
        vector = Helpers().vector2dict(vector)
        vector = vector['expression']
        np.savetxt("./DATASET/expression/{}/eb5_{:06}.txt".format(emotion, i), vector)