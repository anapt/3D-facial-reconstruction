import SemanticCodeVector as scv
import numpy as np


def main():
    # part 1
    path = './DATASET/model2017-1_bfm_nomouth.h5'

    data = scv.SemanticCodeVector(path)
    bases = data.read_pca_bases()

    shape_pca = bases['shape_pca']
    expression_pca = bases['expression_pca']
    x = data.sample_vector()
    vertices = data.calculate_coords(x)
    # sdev = np.std(shape_pca, 0)
    # print(sdev.shape)
    # # print(sdev)
    #
    # shape_pca = np.multiply(shape_pca, np.transpose(sdev))
    # print(shape_pca)
    # print(np.var(shape_pca))
    # print(np.dot(np.transpose(shape_pca), shape_pca))
    # print(np.var(expression_pca))
    # print(np.dot(np.transpose(expression_pca), expression_pca))
    # diag = variance


main()