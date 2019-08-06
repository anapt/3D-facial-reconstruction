import numpy as np
import ImageFormationLayer as ifl
import matplotlib.pyplot as plt
import InverseFaceNetEncoderPredict as net_predict
import cv2
import time


class PrepareImages():
    def __init__(self):
        self.image_dir = './DATASET/face_db/'
        self.net = net_predict.InverseFaceNetEncoderPredict()

    @staticmethod
    def vector_resampling(vector):
        if isinstance(vector, dict):
            pass
        else:
            vector = {
                "shape": np.squeeze(vector[0:80, ]),
                "expression": np.squeeze(vector[80:144, ]),
                "reflectance": np.squeeze(vector[144:224, ]),
                "rotation": np.squeeze(vector[224:227, ]),
                "translation": np.squeeze(vector[227:230, ]),
                "illumination": np.squeeze(vector[230:257, ])
            }

        shape = vector['shape'] + np.random.normal(0, 0.005, 80)

        expression = vector['expression'] + np.random.normal(0, 1, 64)

        reflectance = vector['reflectance'] + np.random.normal(0, 0.2, 80)

        rotation = vector['rotation'] + np.random.uniform(-5, 5, 3)

        translation = np.random.uniform(-0.2, 0.2, 3)
        translation[2] = np.random.normal(0, 0.02, 1)
        translation = vector['translation'] + translation

        illumination = vector['illumination'] + np.random.normal(0, 0.02, 27)

        x = {
            "shape": shape,
            "expression": expression,
            "reflectance": reflectance,
            "rotation": rotation,
            "translation": translation,
            "illumination": illumination
        }

        return x

    def get_prediction(self, image_path):
        vector = self.net.model_predict(image_path=image_path)
        return vector

    def create_image_and_save(self, vector, n):
        # create first image with variation
        x_new = self.vector_resampling(vector)
        formation = ifl.ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(n)
        cv2.imwrite(image_path, image)

        # create second image with variation
        x_new = self.vector_resampling(vector)
        formation = ifl.ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(n+1)
        cv2.imwrite(image_path, image)


def main():
    # Number of images to read
    N = 2
    preprocess = PrepareImages()

    for n in range(0, N):
        start = time.time()
        image_path = preprocess.image_dir + '{}.png'.format(n)
        print(image_path)
        vector = preprocess.get_prediction(image_path)
        preprocess.create_image_and_save(vector, 2*n)
        print("Time passed:", time.time() - start)
        print(n)


main()