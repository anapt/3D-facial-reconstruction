import h5py
import numpy as np
import matplotlib.pyplot as plt


class SemanticCodeVector:

    def __init__(self, path):
        self.model = h5py.File(path, 'r+')

    def read_pca_bases(self):
        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[0:len(shape_pca), 0:80]
        # print(shape_pca.shape)
        # print("shape pca mean")
        # print(np.mean(shape_pca))
        # print(np.std(shape_pca))

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[0:len(reflectance_pca), 0:80]
        # print(reflectance_pca.shape)
        # print("reflectance pca mean")
        # print(np.mean(reflectance_pca))
        # print(np.std(reflectance_pca))

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[0:len(expression_pca), 0:64]
        # print("expression pca mean")
        # print(np.mean(expression_pca))
        # print(expression_pca.shape)

        average_shape = self.model['shape']['model']['mean'][()]
        # print(average_shape.shape)
        # print(type(average_shape))

        average_reflectance = self.model['color']['model']['mean'][()]
        # print(average_reflectance.shape)                        # (159447,)
        # print(type(average_color))                        # <class 'numpy.ndarray'>

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
        # shape -> normal 0,1
        a = np.random.normal(0, 1, 80)

        # reflectance -> normal 0,1
        b = np.random.normal(0, 1, 80)

        # expression -> uniform -12, 12
        d = np.random.uniform(-15, 15, 64)
        # bias of 4.8 to 1st parameter
        d[0] = 100

        # yaw pitch - uniform -40 40
        # roll - uniform -15 15
        rotmat = np.random.uniform(-15, 15, 3)
        rotmat[2] = np.random.uniform(-10, 10, 1)

        # decided after a few tests
        # TODO range is smaller than the one used in the paper
        # illumination parameters uniform -0,2 0,2
        # 1st coefficient uniform 0.6 1.2
        g = np.random.uniform(0.1, 0.3, 27)
        g[0] = np.random.uniform(0.4, 1, 1)
        # TODO add translation
        t = np.zeros(3)

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

        vertices = scv_pca_bases["average_shape"] + \
            np.dot(scv_pca_bases["shape_pca"], vector["shape"]) + \
            np.dot(scv_pca_bases["expression_pca"], vector["expression"])

        print(scv_pca_bases["average_shape"] - vertices)
        return vertices

    def calculate_reflectance(self, vector):
        scv_pca_bases = self.read_pca_bases()

        skin_reflectance = scv_pca_bases["average_reflectance"] + np.dot(scv_pca_bases["reflectance_pca"], vector["reflectance"])

        return skin_reflectance

