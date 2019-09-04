from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pathlib
import cv2
import numpy as np
from prediction_plots import prediction_plots

def main():
    net = InverseFaceNetEncoderPredict()
    n = 11
    path = net.bootstrapping_path + 'MUG/'
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    print(all_image_paths)
    all_image_paths = all_image_paths[0:10]
    for n, path in enumerate(all_image_paths):

        # net.evaluate_model()

        x = net.model_predict(path)
        np.savetxt(net.bootstrapping_path + 'test_loss/x_120_{:06}.txt'.format(n), x)
        x = net.vector2dict(x)

        # x_true = np.loadtxt(net.sem_root + 'training/x_{:06}.txt'.format(n))
        x_true = np.zeros((net.scv_length, ))
        x_true = net.vector2dict(x_true)

        # prediction_plots(x_true, x, save_figs=False)

        image = net.calculate_decoder_output(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(net.bootstrapping_path + 'test_loss/image_120_{:06}.png'.format(n), image)
    # x = np.loadtxt("/home/anapt/data/semantic/x_{:06}.txt".format(8))
    # x = np.loadtxt("./DATASET/semantic/training/x_{:06}.txt".format(1))
    # x = net.vector2dict(x)
    # image = net.calculate_decoder_output(x)
    # show_result = True
    # if show_result:
    #     plt.imshow(image)
    #     plt.show()
    # net.evaluate_model()


main()