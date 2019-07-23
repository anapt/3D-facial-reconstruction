import cv2
import numpy as np


def translate(value, left_min, left_max, right_min=0, right_max=500):
    # Figure out how 'wide' each range is

    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    # print(np.subtract(value, leftMin))
    value_scaled = np.subtract(value, left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


def patch(position, color, cells):
    n_cells = 105694
    w = 500
    image = np.zeros((w, w, 3), dtype=np.uint8)

    for i in range(n_cells-50000, n_cells):
        triangle = cells[:, i]
        # print(i)
        # print(triangle)
        x = position[0, triangle]
        y = position[1, triangle]
        coord = np.transpose(np.vstack((x, y)))
        coord = translate(coord, position.min(), position.max())

        triangle_color = color[:, triangle]
        triangle_color = (np.average(triangle_color, axis=1)) * 255

        cv2.fillPoly(image, np.int32([coord]), color=tuple([int(x) for x in triangle_color]))

    center = (w/2, w/2)
    angle180 = 180
    scale = 1.0

    rotation_matrix = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(image, rotation_matrix, (w, w))

    # cv2.imshow('Image rotated by 180 degrees', rotated180)
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image

    return rotated180
