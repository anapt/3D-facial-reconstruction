import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


class SemanticCodeVector:

    def __init__(self, path):
        self.model = h5py.File(path, 'r+')

    def read_pca_bases(self):
        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[0:len(shape_pca), 0:80]
        sdev = np.std(shape_pca, 0)
        # print("shape variance")
        # print(np.var(shape_pca))
        shape_pca = np.multiply(shape_pca, np.transpose(sdev))
        normalize(shape_pca, copy=False)

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[0:len(reflectance_pca), 0:80]
        sdev = np.std(reflectance_pca, 0)
        reflectance_pca = np.multiply(reflectance_pca, np.transpose(sdev))
        # normalize(reflectance_pca, copy=False)

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), 0:64]
        sdev = np.std(expression_pca, 0)
        expression_pca = np.multiply(expression_pca, np.transpose(sdev))
        normalize(expression_pca, copy=False)

        average_shape = self.model['shape']['model']['mean'][()]

        average_reflectance = self.model['color']['model']['mean'][()]

        scv_pca_bases = {
            "shape_pca": shape_pca,
            "expression_pca": expression_pca,
            "reflectance_pca": reflectance_pca,
            "average_shape": average_shape,
            "average_reflectance": average_reflectance
        }
        return scv_pca_bases

    def read_cells(self):
        cells = self.model['shape']['representer']['cells'][()]
        return cells

    @staticmethod
    def sample_vector():
        a = np.random.normal(0, 1, 80)
        # a = np.random.uniform(-1, 1, 80)

        b = np.random.normal(0, 1, 80)

        # d = np.random.normal(0, 1, 64)
        d = np.random.uniform(-3, 3, 64)
        d[0] = np.random.uniform(-2.5, 2.5, 1)

        rotmat = np.random.uniform(-10, 10, 3)
        rotmat[2] = np.random.uniform(-7, 7, 1)

        # TODO range is smaller than the one used in the paper
        g = np.random.uniform(0.1, 0.4, 27)
        g[0] = np.random.uniform(0.4, 1, 1)

        t = np.random.uniform(-25, 25, 3)

        x = {
            "shape": a,
            "expression": d,
            "reflectance": b,
            "rotation": rotmat,
            "translation": t,
            "illumination": g
        }

        return x

    def plot_face3d(self):

        scv_pca_bases = self.read_pca_bases()
        plot_color = np.asarray(scv_pca_bases["average_reflectance"])
        plot_color = np.reshape(plot_color, (3, int(plot_color.size / 3)), order='F')

        plot_points = np.asarray(scv_pca_bases["average_shape"])
        plot_points = np.reshape(plot_points, (3, int(plot_points.size / 3)), order='F')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.scatter3D(plot_points[0, ], plot_points[1, ], plot_points[2, ], s=1, c=np.transpose(plot_color))

        plt.show()

    def calculate_coords(self, vector):
        scv_pca_bases = self.read_pca_bases()
        # print(vector["shape"])
        vertices = scv_pca_bases["average_shape"] + \
            np.dot(scv_pca_bases["shape_pca"], vector["shape"]) + \
            np.dot(scv_pca_bases["expression_pca"], vector["expression"])

        # print(scv_pca_bases["average_shape"] - vertices)

        return vertices

    def calculate_reflectance(self, vector):
        scv_pca_bases = self.read_pca_bases()

        skin_reflectance = scv_pca_bases["average_reflectance"] +  \
            np.dot(scv_pca_bases["reflectance_pca"], vector["reflectance"])

        # print(scv_pca_bases["average_reflectance"] - skin_reflectance)

        return skin_reflectance

