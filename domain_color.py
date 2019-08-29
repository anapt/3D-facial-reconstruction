from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import cv2

NUM_CLUSTERS = 64

print('reading image')
# im = cv2.imread("./DATASET/bootstrapping/average_color.png", 1)
im = Image.open('./DATASET/bootstrapping/average_color.png')
im = np.array(im, dtype=np.float64) / 255

ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
print(bins.shape)
ind = np.argsort(-counts, axis=0)
# colors = codes[ind[1]]
# print(colors.shape)
average_color = np.ones((5, 5, 3)) * codes[ind[1]]
# average_color = np.reshape(codes[ind[0:25]], newshape=(5, 5, 3), order='A')
# color = np.array((average_color*255), dtype=np.uint8)

# cv2.imshow("", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

print('reading image')
# im = cv2.imread("./DATASET/bootstrapping/average_color.png", 1)
im = Image.open("/home/anapt/Documents/MUG/unpacked/{:06}.jpg".format(1))
im = np.array(im, dtype=np.float64) / 255

ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
print(bins.shape)
ind = np.argsort(-counts, axis=0)
colors = codes[ind[0:25]]
print(colors.shape)

color = np.ones((5, 5, 3)) * codes[ind[1]]
# color = np.reshape(codes[ind[0:25]], newshape=(5, 5, 3), order='A')
# color = np.array((color*255), dtype=np.uint8)

# cv2.imshow("", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

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
dst = im
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
# cv2.imshow("", dst)
cv2.imshow("", cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

# output_image = np.zeros((np.power(dst.shape[0], 2), 3))
#
# output_image[:, 0] = np.reshape(dst[:, :, 0], newshape=(np.power(dst.shape[0], 2), ))
# output_image[:, 1] = np.reshape(dst[:, :, 1], newshape=(np.power(dst.shape[0], 2), ))
# output_image[:, 2] = np.reshape(dst[:, :, 2], newshape=(np.power(dst.shape[0], 2), ))
#
# output_image = np.matmul(output_image, np.linalg.inv(ccm))
#
# dst[:, :, 0] = np.reshape(output_image[:, 0], newshape=(224, 224))
# dst[:, :, 1] = np.reshape(output_image[:, 1], newshape=(224, 224))
# dst[:, :, 2] = np.reshape(output_image[:, 2], newshape=(224, 224))
#
# dst = np.array((dst*255), dtype=np.uint8)
# print(dst.shape)
# cv2.imshow("", dst)
# cv2.waitKey(0)