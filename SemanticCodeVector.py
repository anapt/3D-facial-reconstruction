import h5py
import numpy as np
from sklearn.preprocessing import normalize


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
        average_shape = self.model['shape']['model']['mean'][()]

        average_reflectance = self.model['color']['model']['mean'][()]

        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[0:len(shape_pca), 0:80]

        pca_variance = self.model['shape']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:80]

        sdev = np.sqrt(pca_variance)

        shape_pca = np.multiply(shape_pca, np.transpose(sdev))

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[0:len(reflectance_pca), 0:80]

        pca_variance = self.model['color']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:80]

        sdev = np.sqrt(pca_variance)

        reflectance_pca = np.multiply(reflectance_pca, np.transpose(sdev))

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), 0:64]

        pca_variance = self.model['expression']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:64]

        sdev = np.sqrt(pca_variance)

        expression_pca = np.multiply(expression_pca, np.transpose(sdev))

        scv_pca_bases = {
            "shape_pca": shape_pca,
            "expression_pca": expression_pca,
            "reflectance_pca": reflectance_pca,
            "average_shape": average_shape,
            "average_reflectance": average_reflectance
        }
        return scv_pca_bases

    def get_parameters_dim_sdev(self):
        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[0:len(shape_pca), 0:80]
        shape_sdev = np.std(shape_pca, 0)

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[0:len(reflectance_pca), 0:80]
        reflectance_sdev = np.std(reflectance_pca, 0)

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), 0:64]
        expression_sdev = np.std(expression_pca, 0)

        return shape_sdev, reflectance_sdev, expression_sdev

    def read_cells(self):
        """
        Function that reads vector from .h5py file located in self.path

        :return:
        <class 'numpy.ndarray'> with shape (3, 105694)
        """
        cells = self.model['shape']['representer']['cells'][()]
        return cells

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
        a = np.random.normal(0, 1, 80)

        b = np.random.normal(0, 1, 80)

        d = np.random.normal(0, 1, 64)

        # rotmat = np.random.uniform(-15, 15, 3)
        # rotmat[2] = np.random.uniform(-10, 10, 1)
        rotmat = np.zeros(shape=(3,))
        # TODO range is smaller than the one used in the paper
        g = np.random.uniform(0.2, 0.4, 27)
        g[0] = np.random.uniform(0.2, 0.6, 1)
        # g = np.zeros(shape=(27,))
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
        # vertices = np.dot(scv_pca_bases["shape_pca"], vector["shape"]) + \
        #            np.dot(scv_pca_bases["expression_pca"], vector["expression"])
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

        skin_reflectance = scv_pca_bases["average_reflectance"] +  \
            np.dot(scv_pca_bases["reflectance_pca"], vector["reflectance"])

        return skin_reflectance
