import numpy as np
from FaceNet3D import FaceNet3D as Helpers
from ImageFormationLayer import ImageFormationLayer
import matplotlib.pyplot as plt
import cv2


vector = np.loadtxt("./DATASET/semantic/training/x_{:06}.txt".format(4))
vector = Helpers().vector2dict(vector)
original_expression = vector['expression']
# original = ImageFormationLayer(vector).get_reconstructed_image()
# cv2.imwrite("./DATASET/images/expression_manipulation/happy_original2.png", cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
# # plt.imshow(original)
# # plt.show()
# #
# more_intense = original_expression * 1.5
# more_intense_vector = vector
# more_intense_vector['expression'] = more_intense
# more = ImageFormationLayer(more_intense_vector).get_reconstructed_image()
# cv2.imwrite("./DATASET/images/expression_manipulation/happy_more2.png", cv2.cvtColor(more, cv2.COLOR_BGR2RGB))
# # plt.imshow(more)
# # plt.show()
#
# less_intense = original_expression / 2
# less_intense_vector = vector
# less_intense_vector['expression'] = less_intense
# less = ImageFormationLayer(less_intense_vector).get_reconstructed_image()
# cv2.imwrite("./DATASET/images/expression_manipulation/happy_less2.png", cv2.cvtColor(less, cv2.COLOR_BGR2RGB))
# plt.imshow(less)
# plt.show()
# emotion = 'surprise'
# np.savetxt("./DATASET/expression/{}/ground_truth/center.txt".format(emotion), original_expression)
# zeros = np.zeros((231,))
# zeros = Helpers().vector2dict(zeros)
# zeros['expression'] = original_expression
# original = ImageFormationLayer(zeros).get_reconstructed_image()
# cv2.imwrite("./DATASET/expression/{}/ground_truth/center.png".format(emotion), cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
#
# more_intense = original_expression * 2
# zeros['expression'] = more_intense
# more = ImageFormationLayer(zeros).get_reconstructed_image()
# cv2.imwrite("./DATASET/expression/{}/ground_truth/right_limit.png".format(emotion), cv2.cvtColor(more, cv2.COLOR_BGR2RGB))
#
# less_intense = original_expression / 2
# zeros['expression'] = less_intense
# less = ImageFormationLayer(zeros).get_reconstructed_image()
# cv2.imwrite("./DATASET/expression/{}/ground_truth/left_limit.png".format(emotion), cv2.cvtColor(less, cv2.COLOR_BGR2RGB))

emotion = 'neutral'
center = np.loadtxt('./DATASET/expression/{}/ground_truth/center.txt'.format(emotion))
center[0] = 0
np.savetxt('./DATASET/expression/{}/ground_truth/center2.txt'.format(emotion), center)
# happy = np.loadtxt('./DATASET/expression/{}/ground_truth/happy.txt'.format(emotion))
# mean = center + happy /2
right = center * 2
left = center / 2

# print(np.mean(center)-np.mean(right))
# zeros = np.zeros((231,))
# zeros = Helpers().vector2dict(zeros)
# for i in range(0, 50):
#     noise = np.random.uniform(-0.5, 0.5, 64)
#     zeros['expression'] = center + noise
#     np.savetxt('./DATASET/expression/{}/e_{:06}.txt'.format(emotion, 100+i), zeros['expression'])
#     # img = ImageFormationLayer(zeros).get_reconstructed_image()
#     # cv2.imwrite("./DATASET/expression/{}/im_{:06}.png".format(emotion, i), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
zeros = np.zeros((231,))
zeros = Helpers().vector2dict(zeros)
zeros['expression'] = center
img = ImageFormationLayer(zeros).get_reconstructed_image()
cv2.imwrite("./DATASET/expression/{}/ground_truth/im_{:06}.png".format(emotion, 5), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))