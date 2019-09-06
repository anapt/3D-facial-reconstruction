import numpy as np
import time
import cv2
from ImageFormationLayer import ImageFormationLayer
from LandmarkDetection import LandmarkDetection
from ImagePreprocess import ImagePreprocess
from FaceNet3D import FaceNet3D as Helpers
import pathlib


class LossLayer(Helpers):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    def __init__(self, vector):
        """
                Class initializer
                """
        super().__init__()
        self.x = self.vector2dict(vector)
        self.num_of_vertices = 13000

    def statistical_regularization_term(self):
        weight_expression = 0.8
        weight_reflectance = 1.7e-3
        sr_term = sum(pow(self.x['shape'], 2)) + weight_expression * sum(pow(self.x['expression'], 2)) + \
            weight_reflectance * sum(pow(self.x['color'], 2))

        # print("statistical reg term", sr_term)

        return sr_term

    def dense_photometric_alignment(self, original_image):

        formation = ImageFormationLayer(self.x)
        new_image, indices, position = formation.get_reconstructed_image_for_loss()
        position = self.translate(position, position.min(), position.max(),
                                  right_min=0, right_max=self.IMG_SIZE-1)

        # new_image_aligned = self.align_images(new_image, original_image)
        new_image_aligned = new_image
        photo_term = 0

        for i in range(0, indices.shape[0]):
            photo_term = photo_term + np.linalg.norm(
                original_image[np.int(position[0, indices[i]]), np.int(position[1, indices[i]])] -
                new_image_aligned[np.int(position[0, indices[i]]), np.int(position[1, indices[i]])])

        photo_term = photo_term / indices.shape[0]

        return new_image, photo_term

    @staticmethod
    def sparse_landmark_alignment(original_image, new_image):

        detector = LandmarkDetection()
        # original image landmarks
        landmarks_original = detector.detect_landmarks_for_loss(original_image)
        # reconstructed image landmarks
        landmarks_reconstructed = detector.detect_landmarks_for_loss(new_image)

        alignment_term = (1/46) * pow(np.linalg.norm(landmarks_original - landmarks_reconstructed), 2)

        return alignment_term

    def get_loss(self, original_image):
        weight_photo = 1.92
        weight_reg = 2.9e-5
        weight_land = 1
        new_image, photo_term = self.dense_photometric_alignment(original_image)
        if weight_land == 1:
            alignment_term = self.sparse_landmark_alignment(original_image, new_image)
        else:
            alignment_term = 0

        loss = weight_photo * photo_term + weight_land * alignment_term
            # weight_reg * self.statistical_regularization_term() + \


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


def main():
    n = 1
    # path = Helpers().bootstrapping_path + 'test_loss/'
    # data_root = pathlib.Path(path)
    #
    # mug_path = Helpers().bootstrapping_path + 'MUG/'
    # mug_root = pathlib.Path(mug_path)
    # all_image_paths = list(data_root.glob('*.png'))
    # all_image_paths = [str(path) for path in all_image_paths]
    # all_image_paths.sort()
    # print(all_image_paths[0:10])
    # all_image_paths = all_image_paths[0:10]
    #
    # all_vector_paths = list(data_root.glob('x*.txt'))
    # all_vector_paths = [str(path) for path in all_vector_paths]
    # all_vector_paths.sort()
    # print(all_vector_paths[0:10])
    # all_vector_paths = all_vector_paths[0:10]
    #
    # for n, path in enumerate(all_image_paths):
    #     vector = np.loadtxt(all_vector_paths[n])
    #     ll = LossLayer(vector)
    #
    #     original_image = cv2.imread(path, 1)
    #     # RGB TO BGR
    #     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #
    #     start = time.time()
    #     loss = ll.get_loss(original_image)
    #     # print("Time elapsed: %f " % (time.time() - start))
    #     print("%f" % loss)

    # vector_path = (Helpers().bootstrapping_path + "test_loss/{:06}.txt".format(n))
    # image_path = (Helpers().bootstrapping_path + "test_loss/{:06}.png".format(n))
    #
    # vector = np.loadtxt(vector_path)
    #
    # ll = LossLayer(vector)
    #
    # original_image = cv2.imread(image_path, 1)
    # # RGB TO BGR
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #
    # start = time.time()
    # loss = ll.get_loss(original_image)
    # print("Time elapsed: %f " % (time.time() - start))
    # print("Loss: %f" % loss)

    path = "./DATASET/images/validation/predictions/pimage_{:06}.png".format(1)
    original_image = cv2.imread(path, 1)
    # RGB TO BGR
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    vector = np.loadtxt("./DATASET/images/validation/predictions/x_{:06}.txt".format(0))
    ll = LossLayer(vector)
    start = time.time()
    loss = ll.get_loss(original_image)
    # print("Time elapsed: %f " % (time.time() - start))
    print("%f" % loss)



main()
