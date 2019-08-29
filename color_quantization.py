# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import cv2

n_colors = 81
shp = (9, 9, 3)

# Load the Summer Palace photo
img = cv2.imread("./DATASET/bootstrapping/average_color.png", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

average_color = np.reshape(kmeans.cluster_centers_, newshape=(shp), order='A')
# average_color = np.array((average_color*255), dtype=np.uint8)
#
# cv2.imshow("", cv2.cvtColor(average_color, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

# NEW IMAGE

img = cv2.imread("./DATASET/bootstrapping/MUG/{:06}.png".format(1), 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

color = np.reshape(kmeans.cluster_centers_, newshape=(shp), order='A')
# color = np.array((color*255), dtype=np.uint8)
#
# cv2.imshow("", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))
target_colors = color
source_colors = average_color
target = np.zeros((np.power(target_colors.shape[0], 2), 3))

source = np.zeros((np.power(source_colors.shape[0], 2), 3))
source[:, 0] = np.reshape(source_colors[:, :, 0], newshape=(np.power(source_colors.shape[0], 2), ))
source[:, 1] = np.reshape(source_colors[:, :, 1], newshape=(np.power(source_colors.shape[0], 2), ))
source[:, 2] = np.reshape(source_colors[:, :, 2], newshape=(np.power(source_colors.shape[0], 2), ))

target[:, 0] = np.reshape(target_colors[:, :, 0], newshape=(np.power(source_colors.shape[0], 2), ))
target[:, 1] = np.reshape(target_colors[:, :, 1], newshape=(np.power(source_colors.shape[0], 2), ))
target[:, 2] = np.reshape(target_colors[:, :, 2], newshape=(np.power(source_colors.shape[0], 2), ))


# calculate color correction matrix
ccm = np.matmul(np.linalg.inv(np.matmul(np.transpose(source), source)), (np.matmul(np.transpose(source), target)))
print(ccm)
dst = img
output_image = np.zeros((np.power(dst.shape[0], 2), 3))

output_image[:, 0] = np.reshape(dst[:, :, 0], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 1] = np.reshape(dst[:, :, 1], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 2] = np.reshape(dst[:, :, 2], newshape=(np.power(dst.shape[0], 2), ))

output_image = np.matmul(output_image, ccm)

dst[:, :, 0] = np.reshape(output_image[:, 0], newshape=(224, 224))
dst[:, :, 1] = np.reshape(output_image[:, 1], newshape=(224, 224))
dst[:, :, 2] = np.reshape(output_image[:, 2], newshape=(224, 224))

dst = np.array((dst*255), dtype=np.uint8)
print(dst.shape)
cv2.imshow("", cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

output_image = np.zeros((np.power(dst.shape[0], 2), 3))

output_image[:, 0] = np.reshape(dst[:, :, 0], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 1] = np.reshape(dst[:, :, 1], newshape=(np.power(dst.shape[0], 2), ))
output_image[:, 2] = np.reshape(dst[:, :, 2], newshape=(np.power(dst.shape[0], 2), ))

output_image = np.matmul(output_image, np.linalg.inv(ccm))

dst[:, :, 0] = np.reshape(output_image[:, 0], newshape=(224, 224))
dst[:, :, 1] = np.reshape(output_image[:, 1], newshape=(224, 224))
dst[:, :, 2] = np.reshape(output_image[:, 2], newshape=(224, 224))

dst = np.array((dst*255), dtype=np.uint8)
print(dst.shape)
cv2.imshow("", dst)
cv2.waitKey(0)

# codebook_random = shuffle(image_array, random_state=0)[:n_colors]
# print("Predicting color indices on the full image (random)")
# t0 = time()
# labels_random = pairwise_distances_argmin(codebook_random,
#                                           image_array,
#                                           axis=0)
# print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Display all results, alongside original image
# plt.figure(1)
# plt.clf()
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)
#
# plt.figure(2)
# plt.clf()
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

# plt.figure(3)
# plt.clf()
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(recreate_image(codebook_random, labels_random, w, h))
# plt.show()