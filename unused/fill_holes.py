import cv2
import numpy as np


n = 23
cutout_path = ("./DATASET/images/cutout/im_%d.png" % n)
# read image, ensure binary
image = cv2.imread(cutout_path, 1)
out_face = np.zeros_like(image)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray[gray > 220] = 0
cv2.imshow('', gray)
cv2.waitKey()
# gray = gray.astype(np.bool)
# print(gray)
# gray = np.bitwise_not(gray)
# print(gray)
# out_face[gray] = image[gray]
#
# cv2.imshow('', out_face)
# cv2.waitKey()
# cv2.imwrite(cutout_path, out_face)

# # flood fill background to find inner holes
holes = gray.copy()
cv2.floodFill(holes, None, (0, 0), 255)
cv2.imshow('', holes)
cv2.waitKey()

holes = holes.astype(np.bool)
print(holes)
out_face[holes] = image[holes]
cv2.imshow('', out_face)
cv2.waitKey()
# # # invert holes mask, bitwise or with img fill in holes
# holes = cv2.bitwise_not(holes)
# filled_holes = cv2.bitwise_or(gray, holes)
# cv2.imshow('', filled_holes)
# cv2.waitKey()
