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
from ImageFormationLayer import ImageFormationLayer

archs = {
            "resnet50": 0,
            "boot1": 1,
            "boot2": 2,
            "boot3": 3
        }

stages = list(archs.keys())
stages.sort()


def get_reconstructions(arch):
    path = '/home/anapt/PycharmProjects/thesis/DATASET/real_images/'
    data_root = pathlib.Path(path)

    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:10]

    net = InverseFaceNetEncoderPredict()

    for n, path in enumerate(all_image_paths):
        x = net.model_predict(path)
        np.savetxt("/home/anapt/PycharmProjects/thesis/DATASET/real_images/{}/x_{:06}.txt".format(arch, n), x)
        # image = net.calculate_decoder_output(x)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./DATASET/images/validation/{}/pimg_{:06}.png".format(arch, n), image)


# get_reconstructions(arch='boot3')


def get_loss(arch):
    path = '/home/anapt/PycharmProjects/thesis/DATASET/real_images/'
    data_orig = pathlib.Path(path)

    path = '/home/anapt/PycharmProjects/thesis/DATASET/real_images/{}/'.format(arch)
    data_recon = pathlib.Path(path)

    all_image_paths = list(data_orig.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:5]

    all_vector_paths = list(data_recon.glob('*.txt'))
    all_vector_paths = [str(path) for path in all_vector_paths]
    all_vector_paths.sort()
    # all_vector_paths = all_vector_paths[0:5]

    d = {'{}_loss_real'.format(arch): []}
    df = pd.DataFrame(data=d, dtype=np.float)
    for n, path in enumerate(all_image_paths):
        vector = np.loadtxt(all_vector_paths[n])

        original_image = cv2.imread(path, 1)
        # RGB TO BGR
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        ll = LossLayer(vector)
        loss = ll.get_loss(original_image)

        df = df.append({'{}_loss_real'.format(arch): loss}, ignore_index=True)

    export_csv = df.to_csv(r'./{}_loss_real.csv'.format(arch), index=None, header=True)

    return export_csv


# get_loss(arch='boot1')
# get_loss(arch='boot2')
# get_loss(arch='boot3')


def read_dfs():
    data = pd.read_csv("./EVALUATION/{}_loss_real.csv".format('resnet50'))
    data_boot1 = pd.read_csv("./EVALUATION/{}_loss_real.csv".format('boot1'))
    data_boot2 = pd.read_csv("./EVALUATION/{}_loss_real.csv".format('boot2'))
    data_boot3 = pd.read_csv("./EVALUATION/{}_loss_real.csv".format('boot3'))

    # resnet = data.values
    data_boot1 = data_boot1.values
    data_boot2 = data_boot2.values
    data_boot3 = data_boot3.values
    data.insert(1, "bootstrapping1", data_boot1, True)
    data.insert(2, "bootstrapping2", data_boot2, True)
    data.insert(3, "bootstrapping3", data_boot3, True)

    print(data.head())

    data.to_csv(r'./{}_losses_real.csv'.format('boot'), index=None, header=True)


# read_dfs()

# data = pd.read_csv("./EVALUATION/{}.csv".format('boot_losses_real'))
# print(data.to_latex(index=True, float_format="{:0.3f}".format))


def compare_original_third():
    data = pd.read_csv("./EVALUATION/{}_loss.csv".format('resnet50'))
    data_boot3 = pd.read_csv("./EVALUATION/{}_loss.csv".format('boot3'))
    data_boot3 = data_boot3.values
    data.insert(1, "bootstrapping3", data_boot3, True)

    print(data.head())
    less = data.idxmin(axis=1)
    # print(data.idxmin())
    # print(type(less))
    print(less.value_counts())


compare_original_third()


def get_best_reconstruction():
    data = pd.read_csv("./EVALUATION/{}.csv".format('boot_losses_real'))

    # less = data.idxmin(axis=1)
    less = data.idxmin(axis=1)
    print(data.idxmin())
    # print(type(less))
    print(less.value_counts())


# get_best_reconstruction()


def bar_plot():
    data = pd.read_csv("./EVALUATION/{}.csv".format('boot_losses_real'))
    resnet = np.mean(data['resnet50_loss'])
    boot1 = np.mean(data['bootstrapping1'])
    boot2 = np.mean(data['bootstrapping2'])
    boot3 = np.mean(data['bootstrapping3'])

    means = (boot3, boot2, boot1, resnet)
    # print("{:0.3f}, {:0.3f}".format(means[0], means[3]))
    N = 4
    plt.figure(figsize=(16, 12))
    ind = np.arange(N)  # the x locations for the groups
    width = 0.50  # the width of the bars: can also be len(x) sequence
    std = (np.std(data['bootstrapping3']), np.std(data['bootstrapping2']), np.std(data['bootstrapping1']),
           np.std(data['resnet50_loss']))
    # print("{:0.3f}".format(std))
    # p1 = plt.barh(ind, means, width, xerr=std, tick_label=['ResNet50', '1st Bootstrapping',
    #                                                        '2nd Bootstrapping', '3rd Bootstrapping'],
    #               color=['peachpuff', 'lavender', 'lightcyan', 'mediumaquamarine'])
    p1 = plt.barh(ind, means, width, xerr=std, tick_label=['3rd Bootstrapping',
                                                           '2nd Bootstrapping', '1st Bootstrapping', 'ResNet50'],
                  color=['peachpuff', 'lavender', 'lightcyan', 'mediumaquamarine'])
    plt.xlabel('Loss')

    plt.xticks(np.arange(0, 250, 50))
    plt.ylabel('Bootstrapping iteration', fontsize=20)
    plt.xlabel('Loss', fontsize=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90, ha='center', va='center', fontsize=15)
    # plt.savefig("loss_boot_real.pdf")
    # plt.show()


# bar_plot()

def get_images():
    n = 21
    for arch in stages:
        # vector = np.loadtxt("./DATASET/semantic/validation/{}/x_{:06}.txt".format(arch, n))
        vector = np.loadtxt("./DATASET/real_images/{}/x_{:06}.txt".format(arch, n))
        image = ImageFormationLayer(vector).get_reconstructed_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./DATASET/images/validation/boot_real/{}_{:06}.png'.format(arch, n), image)


# get_images()

