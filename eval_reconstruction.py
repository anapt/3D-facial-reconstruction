import pathlib
import os
import numpy as np
import cv2
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pandas as pd
from LossLayer import LossLayer

architectures = {
    "xception": 0,
    "resnet50":1,
}

def get_reconstructions(arch):
    path = '/home/anapt/PycharmProjects/thesis/DATASET/images/validation/'
    data_root = pathlib.Path(path)

    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:5]

    net = InverseFaceNetEncoderPredict()

    for n, path in enumerate(all_image_paths):
        x = net.model_predict(path)
        np.savetxt("/home/anapt/PycharmProjects/thesis/DATASET/semantic/validation/{}/x_{:06}.txt".format(arch, n), x)
        # image = net.calculate_decoder_output(x)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./DATASET/images/validation/{}/pimg_{:06}.png".format(arch, n), image)


# get_reconstructions(arch='xception')


def get_loss(arch):
    path = '/home/anapt/PycharmProjects/thesis/DATASET/images/validation/'
    data_orig = pathlib.Path(path)

    path = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/validation/{}/'.format(arch)
    data_recon = pathlib.Path(path)

    all_image_paths = list(data_orig.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:5]

    all_vector_paths = list(data_recon.glob('*.txt'))
    all_vector_paths = [str(path) for path in all_vector_paths]
    all_vector_paths.sort()
    # all_vector_paths = all_vector_paths[0:5]

    d = {'{}_loss'.format(arch): []}
    df = pd.DataFrame(data=d, dtype=np.float)
    for n, path in enumerate(all_image_paths):
        vector = np.loadtxt(all_vector_paths[n])

        original_image = cv2.imread(path, 1)
        # RGB TO BGR
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        ll = LossLayer(vector)
        loss = ll.get_loss(original_image)

        df = df.append({'{}_loss'.format(arch): loss}, ignore_index=True)

    export_csv = df.to_csv(r'./{}_loss.csv'.format(arch), index=None, header=True)

    return export_csv


# get_loss(arch='xception')

def read_dfs():
    for arch in architectures:
        print(arch)
    data_resnet = pd.read_csv("./EVALUATION/{}_loss.csv".format('resnet50'))
    # print(data_resnet.head())
    data_xception = pd.read_csv("./EVALUATION/{}_loss.csv".format('xception'))
    # print(data_resnet.head())
    # print(data_resnet['resnet50_loss'][0])

    print(data_resnet.shape)
    count = 0
    for i in range(0, data_resnet.shape[0]):
        if data_resnet['resnet50_loss'][i] > data_xception['xception_loss'][i]:
            count = count + 1

    print(count)

read_dfs()