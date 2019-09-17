import h5py
import numpy as np
from FaceNet3D import FaceNet3D as Helpers


class SemanticCodeVector(Helpers):

    def __init__(self):
        super().__init__()
        self.model = h5py.File(self.path, 'r+')

    def read_pca_bases(self):
        """
        Function that reads vectors from .h5py file located in self.path
        Scales PCA bases with standard deviation.

        :return:
        dictionary with keys    shape_pca               (3*self.num_of_vertices, self.shape_dim)
                                expression_pca          (3*self.num_of_vertices, self.expression_dim)
                                color_pca               (3*self.num_of_vertices, self.color_dim)
                                average_shape           3*self.num_of_vertices,
                                average_color           3*self.num_of_vertices,
        """
        average_shape = self.model['shape']['model']['mean'][()]

        average_color = self.model['color']['model']['mean'][()]

        # read shape pca basis
        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[0:len(shape_pca), 0:self.shape_dim]
        # read shape pca basis variance
        pca_variance = self.model['shape']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.shape_dim]

        # scale basis
        sdev = np.sqrt(pca_variance)
        shape_pca = np.multiply(shape_pca, np.transpose(sdev))

        # read color pca basis
        color_pca = self.model['color']['model']['pcaBasis'][()]
        color_pca = color_pca[0:len(color_pca), 0:self.color_dim]
        # read color pca basis variance
        pca_variance = self.model['color']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.color_dim]

        # scale basis
        sdev = np.sqrt(pca_variance)
        color_pca = np.multiply(color_pca, np.transpose(sdev))

        # read expression pca basis
        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), 0:self.expression_dim]
        # read expression pca basis variance
        pca_variance = self.model['expression']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.expression_dim]

        # scale basis
        sdev = np.sqrt(pca_variance)
        expression_pca = np.multiply(expression_pca, np.transpose(sdev))

        scv_pca_bases = {
            "shape_pca": shape_pca,
            "expression_pca": expression_pca,
            "color_pca": color_pca,
            "average_shape": average_shape,
            "average_color": average_color
        }
        return scv_pca_bases

    def get_bases_std(self):
        """
        Read PCA Variance, get standard devation, scale to belong in (0,1]
        :return:    shape_std               (3*self.num_of_vertices, )
                    expression_std          (3*self.num_of_vertices, )
                    color_std               (3*self.num_of_vertices, )
        """
        # read shape pca basis variance
        pca_variance = self.model['shape']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.shape_dim]

        shape_std = np.sqrt(pca_variance)
        shape_std = shape_std / np.amax(shape_std)

        # read color pca basis variance
        pca_variance = self.model['color']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.color_dim]

        color_std = np.sqrt(pca_variance)
        color_std = color_std / np.amax(color_std)

        # read expression pca basis variance
        pca_variance = self.model['expression']['model']['pcaVariance'][()]
        pca_variance = pca_variance[0:self.expression_dim]

        expression_std = np.sqrt(pca_variance)
        expression_std = expression_std / np.amax(expression_std)

        return shape_std, color_std, expression_std

    def read_cells(self):
        """
        Function that reads vector from .h5py file located in self.path

        :return:
        <class 'numpy.ndarray'> with shape (3, self.num_of_cells)
        """
        cells = self.model['shape']['representer']['cells'][()]
        return cells

    def sample_vector(self):
        """
        Function that samples the semantic code vector

        :return: dictionary with keys       shape           (self.shape_dim,)
                                            expression      (self.expression_dim,)
                                            color           (self.color_dim,)
                                            rotation        (self.rotation_dim,)
        """
        a = np.random.normal(0, 1, self.shape_dim)
        # a = np.zeros((self.shape_dim,))
        b = np.random.normal(0, 1, self.color_dim)
        # b = np.zeros((self.color_dim,))
        d = np.random.normal(0, 1, self.expression_dim)
        # d = np.zeros((self.expression_dim,))
        rotmat = np.random.uniform(-1.5, 1.5, 3)
        rotmat[0] = np.random.uniform(-1.0, 1.0, 1)
        # rotmat = np.array([0, 0, 0])

        x = {
            "shape": a,
            "expression": d,
            "color": b,
            "rotation": rotmat
        }

        return x

    def calculate_3d_vertices(self, vector):
        """
        Calculates the spatial embedding of the vertices based on the PCA bases for shape and expression
        and the average shape of a face and the parameters for shape and expression of the Semantic Code Vector

        :param vector: Semantic Code Vector - only shape and expression parameters are used
        :return: <class 'numpy.ndarray'> with shape (3*self.num_of_vertices,)
        """
        scv_pca_bases = self.read_pca_bases()

        vertices = scv_pca_bases["average_shape"] + \
            np.dot(scv_pca_bases["shape_pca"], vector["shape"]) + \
            np.dot(scv_pca_bases["expression_pca"], vector["expression"])

        return vertices

    def calculate_color(self, vector):
        """
        Calculates the per vertex skin reflectance based on the PCA bases for skin reflectance and the
        average reflectance and the parameters for skin reflectance of the Semantic Code Vector

        :param vector: Semantic Code Vector - only the parameters for reflectance are used
        :return: <class 'numpy.ndarray'> with shape (3*self.num_of_vertices,)
        """
        scv_pca_bases = self.read_pca_bases()

        skin_color = scv_pca_bases["average_color"] +  \
            np.dot(scv_pca_bases["color_pca"], vector["color"])

        return skin_color
