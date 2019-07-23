import numpy as np

import ImageFormationLayer as ifl
import LandmarkDetection as ld
import time
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
        self.num_of_vertices = 53149

    def statistical_regularization_term(self):
        weight_expression = 0.8
        weight_reflectance = 1.7e-3
        sr_term = sum(self.x['shape']) + weight_expression * sum(self.x['expression']) + \
            weight_reflectance * sum(self.x['reflectance'])

        print("statistical reg term", sr_term)
        return sr_term

    def dense_photometric_alignment(self, original_image):

        formation = ifl.ImageFormationLayer(self.x)
        new_image = formation.get_reconstructed_image()

        new_image_aligned = self.align_images(new_image, original_image)

        photo_term = sum(sum(np.linalg.norm(original_image - new_image_aligned, axis=2))) / self.num_of_vertices
        print("photo term", photo_term)

        return new_image_aligned, photo_term

    def sparse_landmark_alignment(self, original_image, new_image):

        detector = ld.LandmarkDetection()
        # original image landmarks
        landmarks_original = detector.detect_landmarks_for_loss(original_image)
        # reconstructed image landmarks
        landmarks_reconstructed = detector.detect_landmarks_for_loss(new_image)

        alignment_term = pow(np.linalg.norm(landmarks_original - landmarks_reconstructed), 2)
        print("alignment term", alignment_term)

        return alignment_term * 0.5

    def get_loss(self, original_image):
        weight_photo = 1.92
        weight_reg = 2.9e-5
        weight_land = 1
        new_image, photo_term = self.dense_photometric_alignment(original_image)
        if weight_land == 1:
            alignment_term = self.sparse_landmark_alignment(original_image, new_image)
        else:
            alignment_term = 0

        loss = weight_photo * photo_term + \
            weight_reg * self.statistical_regularization_term() + \
            weight_land * alignment_term

        print("loss", loss)

        return loss

    def align_images(self, new_image, original_image):
        start = time.time()
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
        end = time.time()
        print(end - start)
        return im1Reg


def main():
    show_result = True
    n = 10
    vector_path = ("./DATASET/semantic/x_%d.txt" % 15)
    image_path = ("./DATASET/images/image_%d.png" % n)
    vector = np.loadtxt(vector_path)

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
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("", original_image)
    # cv2.waitKey(1000)
    # print(ll.get_loss(original_image))
    # print(ll.get_loss(original_image))
    start = time.time()
    ll.get_loss(original_image)
    print("time elapsed = ", time.time() - start)

main()
