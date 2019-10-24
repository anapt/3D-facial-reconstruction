import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers

helper = Helpers()


def prediction_plots(x_true, x, save_figs, path='./plots/'):
    plt.figure()
    plt.title('Shape')
    plt.plot(x['shape'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['shape'], color='blue', marker='+', linestyle='None', alpha=1)
    if save_figs:
        plt.savefig(path + 'shape.png')
    plt.show()

    # plot expression, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Expression')
    plt.plot(x['expression'], color='red', marker='D', linestyle='None')
    plt.plot(x_true['expression'], color='blue', marker='d', linestyle='None', alpha=1)
    if save_figs:
        plt.savefig(path + 'expression.png')
    plt.show()

    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Color')
    plt.plot(x['color'], color='red', marker='h', linestyle='None', alpha=0.5)
    plt.plot(x_true['color'], color='blue', marker='H', linestyle='None', alpha=0.5)
    if save_figs:
        plt.savefig(path + 'color.png')
    plt.show()

    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Rotation')
    plt.plot(x['rotation'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['rotation'], color='blue', marker='+', linestyle='None', alpha=1)
    if save_figs:
        plt.savefig(path + 'rotation.png')
    plt.show()
