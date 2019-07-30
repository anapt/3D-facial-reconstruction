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

    def cutout_mask_array(self, image, n, flip_rgb, save_image):
        """
        Function that:
                        detects landmarks
                        extracts the face
                        gets the mask of the face
                        removes mouth interior
                        (optional) interchanges R and B color channels

        :param image: <class 'numpy.ndarray'> with shape (m, n, 3)
        :param n: number of iteration, used when save_image is True
        :param flip_rgb: boolean: if True return RGB image else return BGR
        :param save_image: boolean if True save image
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
            center = shape[33, :]
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

            out_face = out_face[center[1] - 120:center[1] + 120, center[0] - 120:center[0] + 120]

            if flip_rgb:
                out_face = cv2.cvtColor(out_face, cv2.COLOR_BGR2RGB)

            if save_image:
                cropped_image_path = ("./DATASET/images/image_{:06}.png".format(n))
                cv2.imwrite(cropped_image_path, out_face)

            return out_face

    def detect_landmarks_for_loss(self, image):
        """
        Function that returns Landmark coordinates for original Loss Layer

        :param image:   <class 'numpy.ndarray'> with shape (240, 240, 3)
        :return:        <class 'numpy.ndarray'> with shape (46, 2)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # detect landmarks and transform to image coordinates
        landmarks = self.predictor(gray, dlib.get_rect(gray))
        shape = face_utils.shape_to_np(landmarks)

        # keep only 46 landmarks
        coords = np.array([[shape[17, :], shape[18, :], shape[19, :], shape[20, :], shape[21, :],    # left brow
                           shape[22, :], shape[23, :], shape[24, :], shape[25, :], shape[26, :],    # right brow
                           shape[36, :], shape[39, :], shape[42, :], shape[45, :],  # left and right eye limits
                           shape[27, :], shape[28, :], shape[29, :], shape[30, :], shape[31, :],
                           shape[32, :], shape[33, :], shape[34, :], shape[35, :],  # nose
                           shape[48, :], shape[49, :], shape[50, :], shape[51, :], shape[52, :],
                           shape[53, :], shape[54, :], shape[55, :], shape[56, :], shape[57, :],
                           shape[58, :], shape[59, :], shape[61, :], shape[62, :], shape[63, :],
                           shape[65, :], shape[66, :], shape[67, :],    # mouth
                           shape[6, :], shape[7, :], shape[8, :], shape[9, :], shape[10, :]     # chin
                            ]], dtype=np.int32)

        return coords
