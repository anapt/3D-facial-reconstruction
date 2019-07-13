import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import ImageFormationLayer as ifl

import cv2


class LossLayer:
    PATH = './DATASET/model2017-1_bfm_nomouth.h5'
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    def __init__(self, vector):
        self.x = {
            "shape": vector[0:80, ],
            "expression": vector[80:144, ],
            "reflectance": vector[144:224, ],
            "rotation": vector[224:227, ],
            "translation": vector[227:230, ],
            "illumination": vector[230:257, ]
            }

    def statistical_regularization_term(self):
        weight_expression = 0.8
        weight_reflectance = 1.7e-3
        sr_term = sum(self.x['shape']) + weight_expression * sum(self.x['expression']) + \
            weight_reflectance * sum(self.x['reflectance'])

        return sr_term

    def dense_photometric_alignment(self, original_image):
        # TODO original_image has to come from the tf.dataset
        formation = ifl.ImageFormationLayer(self.PATH, self.x)
        new_image = formation.get_reconstructed_image()
        # plt.imshow(new_image)
        # plt.show()
        # plt.imshow(original_image)
        # plt.show()

        new_image_aligned = self.align_images(new_image, original_image)

        # plt.imshow(new_image_aligned)
        # plt.show()

        # photo_term = sum(sum(np.linalg.norm(original_image - new_image, axis=2))) / 53149
        photo_term = sum(sum(np.linalg.norm(original_image - new_image_aligned, axis=2))) / 53149

        # print("photo term", photo_term)

        return photo_term

    def get_loss(self, original_image):
        weight_photo = 1.92
        weight_reg = 2.9e-5
        # TODO add Sparse Landmark Alignment
        loss = weight_photo * self.dense_photometric_alignment(original_image) + \
            weight_reg * self.statistical_regularization_term()

        # print("loss", loss)

        return loss

    def align_images(self, new_image, original_image):

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(new_image, keypoints1, original_image, keypoints2, matches, None)
        # cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = original_image.shape
        im1Reg = cv2.warpPerspective(new_image, h, (width, height))

        return im1Reg

    # if __name__ == '__main__':
    #     # Read reference image
    #     refFilename = "form.jpg"
    #     print("Reading reference image : ", refFilename)
    #     imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    #
    #     # Read image to be aligned
    #     imFilename = "scanned-form.jpg"
    #     print("Reading image to align : ", imFilename);
    #     im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    #
    #     print("Aligning images ...")
    #     # Registered image will be resotred in imReg.
    #     # The estimated homography will be stored in h.
    #     imReg, h = align_images(im, imReference)
    #
    #     # Write aligned image to disk.
    #     outFilename = "aligned.jpg"
    #     print("Saving aligned image : ", outFilename);
    #     cv2.imwrite(outFilename, imReg)
    #
    #     # Print estimated homography
    #     print("Estimated homography : \n", h)


def main():
    show_result = True
    n = 5
    vector_path = ("./DATASET/semantic/x_%d.txt" % n)
    image_path = ("./DATASET/images/image_%d.png" % 10)
    vector = np.loadtxt(vector_path)
    # print(vector.shape)
    # vector = np.ones((257, ))
    # print(vector.shape)
    x = {
        "shape": vector[0:80, ],
        "expression": vector[80:144, ],
        "reflectance": vector[144:224, ],
        "rotation": vector[224:227, ],
        "translation": vector[227:230, ],
        "illumination": vector[230:257, ]
    }

    ll = LossLayer(vector)
    # sr_term = ll.statistical_regularization_term()
    # print(sr_term)
    original_image = cv2.imread(image_path, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # print(ll.get_loss(original_image))
    # print(ll.get_loss(original_image))
    ll.get_loss(original_image)


main()