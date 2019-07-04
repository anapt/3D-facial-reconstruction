import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_cropper as fc
import background_remover as br


def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image


n = 23
image_path = ("./DATASET/images/im_%d.png" % n)
cutout_path = ("./DATASET/images/cutout/im_%d.png" % n)
cropped_image_path = ("./DATASET/images/cropped/image_%d.png" % n)
image = cv2.imread(image_path, 1)

out_face = np.zeros_like(image)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./DATASET/shape_predictor_68_face_landmarks.dat")

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in bw image
faces = detector(gray)
for face in faces:

    landmarks = predictor(gray, face)
    shape = face_utils.shape_to_np(landmarks)

    # for n in range(60, 68):
    #     x = landmarks.part(n).x
    #     y = landmarks.part(n).y
    #     cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    #
    # cv2.imwrite('im.png', image)

    # initialize mask array
    remapped_shape = np.zeros_like(shape)
    feature_mask = np.zeros((image.shape[0], image.shape[1]))

    # we extract the face
    remapped_shape = face_remap(shape)
    # get the mask of the face
    cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
    feature_mask = feature_mask.astype(np.bool)

    out_face[feature_mask] = image[feature_mask]
    cv2.imwrite(cutout_path, out_face)

    # detector = fc.FaceCropper()
    # detector.generate(cutout_path, cropped_image_path, False, True)



# cv2.imshow("Frame", cap)



# cutout = cv2.add(cap, landmarks)

# cv2.imwrite(cutout_path, cap)
