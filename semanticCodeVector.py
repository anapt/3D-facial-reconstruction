import h5py
import numpy as np
import matplotlib.pyplot as plt



class SemanticCodeVector:

    def __init__(self, path):
        self.model = h5py.File(path, 'r+')

    def read_pca_bases(self):
        shape_pca = self.model['shape']['model']['pcaBasis'][()]
        shape_pca = shape_pca[1:len(shape_pca), 0:80]
        # print(shape_pca.shape)

        reflectance_pca = self.model['color']['model']['pcaBasis'][()]
        reflectance_pca = reflectance_pca[1:len(reflectance_pca), 0:80]
        # print(reflectance_pca.shape)

        expression_pca = self.model['expression']['model']['pcaBasis'][()]
        expression_pca = expression_pca[1:len(expression_pca), 0:64]
        # print(expression_pca.shape)

        average_shape = self.model['shape']['model']['mean'][()]
        # print(average_shape.shape)
        # print(type(average_shape))

        average_reflectance = self.model['color']['model']['mean'][()]
        # print(average_color.shape)                        # (159447,)
        # print(type(average_color))                        # <class 'numpy.ndarray'>

        scv_pca_bases = {
            "shape_pca": shape_pca,
            "expression_pca": expression_pca,
            "reflectance_pca": reflectance_pca,
            "average_shape": average_shape,
            "average_reflectance": average_reflectance
        }
        return scv_pca_bases

    def sample_vector(self):
        a = np.random.normal(0, 1, 80)
        b = np.random.normal(0, 1, 80)
        d = np.random.uniform(-24, 24, 64)
        d[0] = np.random.uniform(-4.8, 4.8, 1)

        rotmat = np.random.uniform(-40, 40, 3)
        rotmat[2] = np.random.uniform(-15, 15, 1)

        g = np.random.uniform(-0.2, 0.2, 27)
        g[0] = np.random.uniform(0.6, 1.2, 1)

        x = {
            "shape": a,
            "expression": d,
            "skin_reflectance": b,
            "rotation": rotmat,
            # "translation": t,
            "scene_illumination": g
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


def main():

    # MODIFY TO path containing Basel Face model
    path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
    scv = SemanticCodeVector(path)
    # x = scv.sample_vector()

    # print(x["scene_illumination"])

    scv.plot_face3d()


main()
