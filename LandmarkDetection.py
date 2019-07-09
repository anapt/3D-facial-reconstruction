import cv2
import numpy as np
import dlib
from imutils import face_utils


class LandmarkDetection:
    PREDICTOR_PATH = "/home/anapt/PycharmProjects/thesis/DATASET/shape_predictor_68_face_landmarks.dat"

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    @staticmethod
    def face_remap(shape):
        remapped_image = cv2.convexHull(shape)
        return remapped_image

    def cutout_mask(self, image_path, cutout_path):
        image = cv2.imread(image_path, 1)
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

            mouth = np.array([[shape[60, :], shape[61, :], shape[62, :], shape[63, :], shape[64, :],
                               shape[65, :], shape[66, :], shape[67, :]]], dtype=np.int32)

            cv2.fillConvexPoly(feature_mask, mouth, 0)

            feature_mask = feature_mask.astype(np.bool)

            out_face[feature_mask] = image[feature_mask]

            cv2.imwrite(cutout_path, out_face)

            return out_face

    def cutout_mask_array(self, image):
        out_face = np.zeros_like(image)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        print(type(gray))
        # detect faces in bw image
        faces = self.detector(np.uint8(gray))
        for face in faces:
            landmarks = self.predictor(np.uint8(gray), face)
            # print(landmarks)
            shape = face_utils.shape_to_np(landmarks)
            # initialize mask array
            # remapped_shape = np.zeros_like(shape)
            feature_mask = np.zeros((image.shape[0], image.shape[1]))

            # we extract the face
            remapped_shape = self.face_remap(shape)
            # get the mask of the face
            cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)

            mouth = np.array([[shape[60, :], shape[61, :], shape[62, :], shape[63, :], shape[64, :],
                               shape[65, :], shape[66, :], shape[67, :]]], dtype=np.int32)

            cv2.fillConvexPoly(feature_mask, mouth, 0)

            feature_mask = feature_mask.astype(np.bool)

            out_face[feature_mask] = image[feature_mask]

            return out_face

    @staticmethod
    def remove_mouth(image, cutout_path):
        out_face = np.zeros_like(image)
        # convert to grayscale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # threshold
        gray[gray > 230] = 0

        # flood fill background to find inner holes
        holes = gray.copy()
        cv2.floodFill(holes, None, (0, 0), 255)

        holes = holes.astype(np.bool)
        out_face[holes] = image[holes]

        cv2.imwrite(cutout_path, out_face)
