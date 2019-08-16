import numpy as np
import matplotlib.pyplot as plt
from refactor.FaceNet3D import FaceNet3D as Helpers

helper = Helpers()


def main():
    x_true = np.loadtxt(helper.sem_root + 'training/x_{:06}.txt'.format(2))
    x_true = helper.vector2dict(x_true)

    x = np.loadtxt('/home/anapt/PycharmProjects/thesis/refactor/x_boot.txt')
    x = helper.vector2dict(x)

    # x = x_true
    # print(x['translation'])
    # print(sum(abs(x['shape'])))
    # print(sum(abs(x['expression'])))
    # print(sum(abs(x['reflectance'])))
    # print(sum(abs(x['rotation'])))
    # print(sum(abs(x['translation'])))
    # print(sum(abs(x['illumination'])))
    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Shape')
    plt.plot(x['shape'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['shape'], color='blue', marker='+', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/shape.png')
    plt.show()

    # plot expression, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Expression')
    plt.plot(x['expression'], color='red', marker='D', linestyle='None')
    plt.plot(x_true['expression'], color='blue', marker='d', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/expression.png')
    plt.show()

    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Color')
    plt.plot(x['color'], color='red', marker='h', linestyle='None', alpha=0.5)
    plt.plot(x_true['color'], color='blue', marker='H', linestyle='None', alpha=0.5)
    plt.savefig('./tests/exp7/color.png')
    plt.show()

    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Rotation')
    plt.plot(x['rotation'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['rotation'], color='blue', marker='+', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/rotation.png')
    plt.show()



main()