import pathlib
import os
import numpy as np
import cv2
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pandas as pd
import seaborn
from LossLayer import LossLayer
import matplotlib.pyplot as plt
from prediction_plots import prediction_plots
from FaceNet3D import FaceNet3D as Helpers

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
# print(data.to_latex(index=True))


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


# get_best_reconstruction()


def get_plots():
    path = './DATASET/semantic/validation/{}/x_{:06}.txt'.format('inceptionV3', 2)
    inception = np.loadtxt(path)
    inception = Helpers().vector2dict(inception)

    path = './DATASET/semantic/validation/{}/x_{:06}.txt'.format('resnet50', 2)
    resnet = np.loadtxt(path)
    resnet = Helpers().vector2dict(resnet)

    path = './DATASET/semantic/validation/{}/x_{:06}.txt'.format('xception', 2)
    xception = np.loadtxt(path)
    xception = Helpers().vector2dict(xception)

    path = './DATASET/semantic/validation/x_{:06}.txt'.format(2)
    true = np.loadtxt(path)
    true = Helpers().vector2dict(true)

    plt.figure()
    plt.title('Shape')
    plt.plot(inception['shape'], color='mediumaquamarine', marker='h', linestyle='None', alpha=1, label='InceptionV3')
    plt.plot(resnet['shape'], color='goldenrod', marker='h', linestyle='None', alpha=1, label='ResNet50')
    plt.plot(xception['shape'], color='forestgreen', marker='h', linestyle='None', alpha=1, label='Xception')
    plt.plot(true['shape'], color='firebrick', marker='H', linestyle='None', alpha=1, label='Original')
    # if save_figs:
    #     plt.savefig(path + 'shape.png')
    plt.yticks(rotation=0, fontsize=7)
    plt.xticks(rotation=0, fontsize=7)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    # plt.show()
    plt.savefig('./EVALUATION/shape.png')

    plt.figure()
    plt.title('Expression')
    plt.plot(inception['expression'], color='mediumaquamarine', marker='h', linestyle='None', alpha=1, label='InceptionV3')
    plt.plot(resnet['expression'], color='goldenrod', marker='h', linestyle='None', alpha=1, label='ResNet50')
    plt.plot(xception['expression'], color='forestgreen', marker='h', linestyle='None', alpha=1, label='Xception')
    plt.plot(true['expression'], color='firebrick', marker='H', linestyle='None', alpha=1, label='Original')
    # if save_figs:
    #     plt.savefig(path + 'shape.png')
    plt.yticks(rotation=0, fontsize=7)
    plt.xticks(rotation=0, fontsize=7)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    # plt.show()
    plt.savefig('./EVALUATION/expression.png')

    plt.figure()
    plt.title('Rotation')
    plt.plot(inception['rotation'], color='mediumaquamarine', marker='h', linestyle='None', alpha=1, label='InceptionV3')
    plt.plot(resnet['rotation'], color='goldenrod', marker='h', linestyle='None', alpha=1, label='ResNet50')
    plt.plot(xception['rotation'], color='forestgreen', marker='h', linestyle='None', alpha=1, label='Xception')
    plt.plot(true['rotation'], color='firebrick', marker='H', linestyle='None', alpha=1, label='Original')
    # if save_figs:
    #     plt.savefig(path + 'shape.png')
    plt.yticks(rotation=0, fontsize=7)
    plt.xticks(rotation=0, fontsize=7)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    # plt.show()
    plt.savefig('./EVALUATION/rotation.png')

    plt.figure()
    plt.title('Color')
    plt.plot(inception['color'], color='mediumaquamarine', marker='h', linestyle='None', alpha=1, label='InceptionV3')
    plt.plot(resnet['color'], color='goldenrod', marker='h', linestyle='None', alpha=1, label='ResNet50')
    plt.plot(xception['color'], color='forestgreen', marker='h', linestyle='None', alpha=1, label='Xception')
    plt.plot(true['color'], color='firebrick', marker='H', linestyle='None', alpha=1, label='Original')
    # if save_figs:
    #     plt.savefig(path + 'shape.png')
    plt.yticks(rotation=0, fontsize=7)
    plt.xticks(rotation=0, fontsize=7)
    plt.legend(loc='upper left', shadow=True, fontsize='small')
    # plt.show()
    plt.savefig('./EVALUATION/color.png')


# get_plots()
