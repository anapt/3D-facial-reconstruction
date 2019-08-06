import numpy as np
import ImageFormationLayer as ifl
import matplotlib.pyplot as plt
from model_predict import InverseFaceNetPredict


net = InverseFaceNetPredict()
# net.evaluate_model()
image_path = './landmark_cut.png'

x = net.model_predict(image_path)
np.savetxt("./x_boot.txt", x)


show_result = True
formation = ifl.ImageFormationLayer(x)
image = formation.get_reconstructed_image()
print(image.shape)

if show_result:
    plt.imshow(image)
    plt.show()
