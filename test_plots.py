import numpy as np
import matplotlib.pyplot as plt


def vector2dict(vector):
    """
    Method that transforms (257,) nd.array to dictionary

    :param vector: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
    :return:
    dictionary with keys    shape           (80,)
                            expression      (64,)
                            reflectance     (80,)
                            rotation        (3,)
                            translation     (3,)
                            illumination    (27,)
    """
    x = {
        "shape": np.squeeze(vector[0:80, ]),
        "expression": np.squeeze(vector[80:144, ]),
        "reflectance": np.squeeze(vector[144:224, ]),
        "rotation": np.squeeze(vector[224:227, ]),
        "translation": np.squeeze(vector[227:230, ]),
        "illumination": np.squeeze(vector[230:257, ])
    }
    return x


def main():
    x_true = np.loadtxt('./DATASET/semantic/over/x_{:06}.txt'.format(0))
    x_true = vector2dict(x_true)

    x = np.loadtxt('./x_pred.txt')
    x = vector2dict(x)

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
    plt.title('Reflectance')
    plt.plot(x['reflectance'], color='red', marker='h', linestyle='None', alpha=0.5)
    plt.plot(x_true['reflectance'], color='blue', marker='H', linestyle='None', alpha=0.5)
    plt.savefig('./tests/exp7/reflectance.png')
    plt.show()

    # plot shape, red dots are predicted values
    # blue crosses are ground truth
    plt.figure()
    plt.title('Rotation')
    plt.plot(x['rotation'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['rotation'], color='blue', marker='+', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/rotation.png')
    plt.show()

    plt.figure()
    plt.title('Translation')
    plt.plot(x['translation'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['translation'], color='blue', marker='+', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/translation')
    plt.show()

    plt.figure()
    plt.title('Illumination')
    plt.plot(x['illumination'], color='red', marker='o', linestyle='None')
    plt.plot(x_true['illumination'], color='blue', marker='+', linestyle='None', alpha=1)
    plt.savefig('./tests/exp7/illumination.png')
    plt.show()



main()