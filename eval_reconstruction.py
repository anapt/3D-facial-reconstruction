import pathlib
import os
import numpy as np
import cv2
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pandas as pd
import seaborn
from LossLayer import LossLayer
import matplotlib.pyplot as plt

architectures = {
    "xception": 0,
    "resnet50": 1,
    "inceptionV3": 2
}


def get_reconstructions(arch):
    path = '/home/anapt/PycharmProjects/thesis/DATASET/images/validation/'
    data_root = pathlib.Path(path)

    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:10]

    net = InverseFaceNetEncoderPredict()

    for n, path in enumerate(all_image_paths):
        x = net.model_predict(path)
        np.savetxt("/home/anapt/PycharmProjects/thesis/DATASET/semantic/validation/{}/x_{:06}.txt".format(arch, n), x)
        # image = net.calculate_decoder_output(x)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./DATASET/images/validation/{}/pimg_{:06}.png".format(arch, n), image)


# get_reconstructions(arch='inceptionV3')


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


# get_loss(arch='inceptionV3')


def read_dfs():
    for arch in architectures:
        print(arch)
    data_resnet = pd.read_csv("./EVALUATION/{}_loss.csv".format('resnet50'))
    data_xception = pd.read_csv("./EVALUATION/{}_loss.csv".format('xception'))
    data_inception = pd.read_csv("./EVALUATION/{}_loss.csv".format('inceptionV3'))

    resnet = data_resnet.values
    inception = data_inception.values
    data_xception.insert(1, "resnet50_loss", resnet, True)
    data_xception.insert(2, "inceptionV3_loss", inception, True)
    print(data_xception.head())

    count = 0

    data_xception.to_csv(r'./{}_losses.csv'.format('all'), index=None, header=True)
    # for i in range(0, data_resnet.shape[0]):
        # if data_resnet['resnet50_loss'][i] > data_xception['xception_loss'][i]:
        #     count = count + 1

    print(count)

# read_dfs()

# data = pd.read_csv("./EVALUATION/{}.csv".format('all_losses'))
# print(data.to_latex(index=False))


def bar_plot():
    data = pd.read_csv("./EVALUATION/{}.csv".format('all_losses'))
    mean_xception = np.mean(data['xception_loss'])
    mean_resnet = np.mean(data['resnet50_loss'])
    mean_inception = np.mean(data['inceptionV3_loss'])

    means = (mean_inception, mean_resnet, mean_xception)
    N = 3
    plt.figure(figsize=(16, 9))
    ind = np.arange(N)  # the x locations for the groups
    width = 0.40  # the width of the bars: can also be len(x) sequence
    std = (np.std(data['xception_loss']), np.std(data['resnet50_loss']), np.std(data['inceptionV3_loss']))
    p1 = plt.barh(ind, means, width, xerr=std, tick_label=['InceptionV3', 'ResNet50', 'Xception'],
                  color=['peachpuff', 'lavender', 'lightcyan'])

    plt.xlabel('Loss')
    # plt.title('Difference between original and reconstructed faces', fontsize=20)
    # plt.yticks(ind, ('Xception', 'ResNet50', 'InceptionV3'))
    plt.xticks(np.arange(0, 300, 50))
    # plt.title("Confusion Matrix", fontsize=20)
    plt.ylabel('Network Architecture', fontsize=20)
    plt.xlabel('Loss', fontsize=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90, ha='center', va='center', fontsize=15)
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.savefig("loss.pdf")
    # plt.show()


# bar_plot()


def get_best_reconstruction():
    data = pd.read_csv("./EVALUATION/{}.csv".format('all_losses'))

    less = data.idxmin(axis=1)
    # print(type(less))
    print(less.value_counts())


get_best_reconstruction()
