import cv2
import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers


path = "/home/anapt/Documents/MUG/cropped/{:06}.png".format(0)
img = cv2.imread(path, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# image = InverseFaceNetEncoderPredict().calculate_decoder_output(x)

kernel = np.ones((3, 3), np.float32)/9
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()

# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
source_colors = dst[90:102, 90:102, :]

cv2.imshow("", cv2.cv2.cvtColor(source_colors, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

# convert to float
source_colors = source_colors/255.0

source = np.zeros((np.power(source_colors.shape[0], 2), 3))
source[:, 0] = np.reshape(source_colors[:, :, 0], newshape=(144, ))
source[:, 1] = np.reshape(source_colors[:, :, 1], newshape=(144, ))
source[:, 2] = np.reshape(source_colors[:, :, 2], newshape=(144, ))


img = cv2.imread("./DATASET/images/limits/color/0.png", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
target_colors = img[224:236, 224:236, :]

cv2.imshow("", cv2.cv2.cvtColor(target_colors, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

# convert to float
target_colors = target_colors/255.0

target = np.zeros((np.power(target_colors.shape[0], 2), 3))
target[:, 0] = np.reshape(target_colors[:, :, 0], newshape=(144, ))
target[:, 1] = np.reshape(target_colors[:, :, 1], newshape=(144, ))
target[:, 2] = np.reshape(target_colors[:, :, 2], newshape=(144, ))

print(target.shape)
# print(target_colors.shape)

# calculate color correction matrix
ccm = np.matmul(np.linalg.inv(np.matmul(np.transpose(source), source)), (np.matmul(np.transpose(source), target)))
print(ccm)

output_image = np.zeros((np.power(dst.shape[0], 2), 3))
dst = dst/255.0

output_image[:, 0] = np.reshape(dst[:, :, 0], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 1] = np.reshape(dst[:, :, 1], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 2] = np.reshape(dst[:, :, 2], newshape=(np.power(dst.shape[0], 2), ))

output_image = np.matmul(output_image, ccm)

dst[:, :, 0] = np.reshape(output_image[:, 0], newshape=(224, 224))
dst[:, :, 1] = np.reshape(output_image[:, 1], newshape=(224, 224))
dst[:, :, 2] = np.reshape(output_image[:, 2], newshape=(224, 224))

dst = np.array((dst*255), dtype=np.uint8)
print(dst.shape)
cv2.imshow("", cv2.cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

cv2.imshow("", dst)
cv2.waitKey(0)






# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
#
# img = cv2.imread("./DATASET/images/training/image_{:06}.png".format(1), 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

# show_result = True
# if show_result:
#     plt.imshow(image)
#     plt.show()
