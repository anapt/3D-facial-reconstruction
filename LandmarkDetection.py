import cv2
import numpy as np
import dlib
from imutils import face_utils


class LandmarkDetection:
    PREDICTOR_PATH = "/home/anapt/PycharmProjects/thesis/DATASET/shape_predictor_68_face_landmarks.dat"

    def __init__(self):
        """
        Class initializer, initialize detector and landmark predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    @staticmethod
    def face_remap(shape):
        """
        extracts convex part of the face from the landmarks detected

        :param shape: <class 'numpy.ndarray'> with shape (68, 2)
        :return: convex part between landmarks 0:27, <class 'numpy.ndarray'> with shape (20, 1, 2)
        """
        remapped_image = cv2.convexHull(shape)
        return remapped_image

    def cutout_mask_array(self, image, flip_rgb):
        """
        Function that:
                        detects landmarks
                        extracts the face
                        gets the mask of the face
                        removes mouth interior
                        (optional) interchanges R and B color channels

        :param image: <class 'numpy.ndarray'> with shape (m, n, 3)
        :param flip_rgb: boolean: if True return RGB image else return BGR
        :return: mask of the face without the mouth interior <class 'numpy.ndarray'> with shape (m, n, 3)
        """
        out_face = np.zeros_like(image)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in bw image
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            # print(landmarks)
            shape = face_utils.shape_to_np(landmarks)
            # initialize mask array
            # remapped_shape = np.zeros_like(shape)
            feature_mask = np.zeros((image.shape[0], image.shape[1]))

            # we extract the face
            remapped_shape = self.face_remap(shape)
            # get the mask of the face
            cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)

            # extract the mouth
            mouth = np.array([[shape[60, :], shape[61, :], shape[62, :], shape[63, :], shape[64, :],
                               shape[65, :], shape[66, :], shape[67, :]]], dtype=np.int32)
            # remove mouth interior
            cv2.fillConvexPoly(feature_mask, mouth, 0)

            feature_mask = feature_mask.astype(np.bool)

            out_face[feature_mask] = image[feature_mask]

            if flip_rgb:
                out_face = cv2.cvtColor(out_face, cv2.COLOR_BGR2RGB)

            return out_face

    def detect_landmarks_for_loss(self, image):
        out_face = np.zeros_like(image)
        print("hey")
        # cv2.imshow("", image)
        # cv2.waitKey(1000)
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imshow("", gray)
        cv2.waitKey(1000)
        # detect faces in bw image
        landmarks = self.predictor(gray, dlib.get_rect(gray))
        print(landmarks)
        shape = face_utils.shape_to_np(landmarks)
        print("shape", shape)
        # # initialize mask array
        # # remapped_shape = np.zeros_like(shape)
        # feature_mask = np.zeros((image.shape[0], image.shape[1]))
        #
        # # we extract the face
        # remapped_shape = self.face_remap(shape)
        # # get the mask of the face
        # cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
        #
        # # extract the mouth
        # mouth = np.array([[shape[60, :], shape[61, :], shape[62, :], shape[63, :], shape[64, :],
        #                    shape[65, :], shape[66, :], shape[67, :]]], dtype=np.int32)
        # # remove mouth interior
        # cv2.fillConvexPoly(feature_mask, mouth, 0)
        #
        # feature_mask = feature_mask.astype(np.bool)
        #
        # out_face[feature_mask] = image[feature_mask]

        return out_face
