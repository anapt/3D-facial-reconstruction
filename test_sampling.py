import h5py
import numpy as np
from sklearn.preprocessing import normalize
import ParametricMoDecoder as pmd

class SemanticCodeVector:

    def __init__(self, path):
        """
        Class initializer

        :param path: path to Basel Face Model
        """
        self.model = h5py.File(path, 'r+')

    def read_pca_bases(self):
        """
        Function that reads vectors from .h5py file located in self.path

        :return:
        dictionary with keys    shape_pca               159447 80
                                expression_pca          159447 64
                                reflectance_pca         159447 80
                                average_shape           159447
                                average_reflectance     159447
        """

        shape_pca = self.model['shape']['model']['pcaBasis'][()]

        shape_pca = shape_pca[0:len(shape_pca), :]
        # sdev = np.std(shape_pca, 0)
        #
        # shape_pca = np.multiply(shape_pca, np.transpose(sdev))
        # normalize(shape_pca, copy=False, norm='l1')

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[0:len(reflectance_pca), :]


        # normalize(reflectance_pca, copy=False, norm='l1')

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), :]
        print(expression_pca.shape)
        # sdev = np.std(expression_pca, 0)
        # expression_pca = np.multiply(expression_pca, np.transpose(sdev))
        # normalize(expression_pca, copy=False, norm='l1')

        average_shape = self.model['shape']['model']['mean'][()]
        np.savetxt("./avg_shape.txt", average_shape)
        average_reflectance = self.model['color']['model']['mean'][()]
        np.savetxt("./avg_reflectance.txt", average_reflectance)
        scv_pca_bases = {
            "shape_pca": shape_pca,
            "expression_pca": expression_pca,
            "reflectance_pca": reflectance_pca,
            "average_shape": average_shape,
            "average_reflectance": average_reflectance
        }
        return scv_pca_bases

    @staticmethod
    def sample_vector():
        """
        Function that samples the semantic code vector

        :return:
        dictionary with keys    shape           (80,)
                                expression      (64,)
                                reflectance     (80,)
                                rotation        (3,)
                                translation     (3,)
                                illumination    (27,)
        """
        # a = np.random.normal(0, 1, 80)
        a = np.zeros(shape=(199,))
        # a = np.random.uniform(-4, 4, 80)
        # a = 1000 * a

        # b = np.random.normal(0, 0.15, 80)
        b = np.zeros(shape=(199,))
        # b[0] = 2000
        # d = np.random.normal(0, 1, 64)
        # d = np.random.uniform(-14, 14, 64)
        # d[0] = 10*d[0]
        d = np.zeros(shape=(100,))
        d[0] = 2000
        # rotmat = np.random.uniform(-15, 15, 3)
        # rotmat[2] = np.random.uniform(-10, 10, 1)
        rotmat = np.zeros(shape=(3,))
        # TODO range is smaller than the one used in the paper
        # g = np.random.uniform(0.2, 0.4, 27)
        # g[0] = np.random.uniform(0.5, 1, 1)
        g = np.ones(shape=(27,))
        t = np.ones(shape=(3,)) * 2.5
        t[2] = -50
        # t = np.random.uniform(1.50, 3.50, 3)
        # t[2] = np.random.uniform(-0.30, 0.30, 1)

        x = {
            "shape": a,
            "expression": d,
            "reflectance": b,
            "rotation": rotmat,
            "translation": t,
            "illumination": g
        }

        return x

    def calculate_coords(self, vector):
        """
        Calculates the spatial embedding of the vertices based on the PCA bases for shape and expression
        and the average shape of a face and the parameters for shape and expression of the Semantic Code Vector

        :param vector: Semantic Code Vector - only shape and expression parameters are used
        :return: <class 'numpy.ndarray'> with shape (159447,)
        """
        scv_pca_bases = self.read_pca_bases()
        # print(vector["shape"])
        vertices = scv_pca_bases["average_shape"] + \
                   np.dot(scv_pca_bases["shape_pca"], vector["shape"]) + \
                   np.dot(scv_pca_bases["expression_pca"], vector["expression"])
        # print(vertices)
        return vertices

    def calculate_reflectance(self, vector):
        """
        Calculates the per vertex skin reflectance based on the PCA bases for skin reflectance and the
        average reflectance and the parameters for skin reflectance of the Semantic Code Vector

        :param vector: Semantic Code Vector - only the parameters for reflectance are used
        :return: <class 'numpy.ndarray'> with shape (159447,)
        """
        scv_pca_bases = self.read_pca_bases()

        skin_reflectance = scv_pca_bases["average_reflectance"] + \
                           np.dot(scv_pca_bases["reflectance_pca"], vector["reflectance"])

        return skin_reflectance


def main():
    data = SemanticCodeVector('./DATASET/model2017-1_bfm_nomouth.h5')
    vector = data.sample_vector()
    vertices = data.calculate_coords(vector)
    reflectance = data.calculate_reflectance(vector)

    ws_vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
    print(ws_vertices)
    reflectance = np.reshape(reflectance, (3, int(reflectance.size / 3)), order='F')
    print(np.ceil(reflectance * 255))


main()