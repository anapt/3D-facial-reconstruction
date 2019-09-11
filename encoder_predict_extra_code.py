from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from prediction_plots import prediction_plots


def main():
    net = InverseFaceNetEncoderPredict()
    n = 11
    # path = net.bootstrapping_path + 'MUG/'
    path = "./DATASET/images/validation/"
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('image_01*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    print(all_image_paths)
    all_image_paths = all_image_paths[0:5]
    for n, path in enumerate(all_image_paths):

        # net.evaluate_model()

        x = net.model_predict(path)
        # np.savetxt(net.bootstrapping_path + 'test_loss/x_120_{:06}.txt'.format(n), x)
        np.savetxt("./DATASET/images/validation/pb1/x_{:06}.txt".format(n+16), x)
        x = net.vector2dict(x)

        x_true = np.loadtxt(net.sem_root + 'validation/x_{:06}.txt'.format(n))
        # x_true = np.zeros((net.scv_length, ))
        x_true = net.vector2dict(x_true)

        prediction_plots(x_true, x, save_figs=False)

        image = net.calculate_decoder_output(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(net.bootstrapping_path + 'test_loss/image_120_{:06}.png'.format(n), image)
        cv2.imwrite("./DATASET/images/validation/pb1/pimage_{:06}.png".format(n+16), image)
    # x = np.loadtxt("./DATASET/images/validation/p240/xx_{:06}.txt".format(43))
    # x_target = np.loadtxt("./DATASET/images/validation/p240/xx_{:06}.txt".format(2))
    # x = net.vector2dict(x)
    # x_target = net.vector2dict(x_target)
    # x_target['expression'] = x['expression']
    #
    # image = net.calculate_decoder_output(x_target)
    # cv2.imwrite("expression_transfer.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # show_result = False
    # if show_result:
    #     plt.imshow(image)
    #     plt.show()
    # net.evaluate_model()


main()