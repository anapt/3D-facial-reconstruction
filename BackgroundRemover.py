import cv2
import numpy as np
import sys
import os


class BackgroundRemover(object):

    def __init__(self):
        self.edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("./DATASET/model.yml")

    @staticmethod
    def filter_out_salt_pepper_noise(edge_img):
        # Get rid of salt & pepper noise.
        count = 0
        last_median = edge_img
        median = cv2.medianBlur(edge_img, 3)
        while not np.array_equal(last_median, median):
            # get those pixels that gets zeroed out
            zeroed = np.invert(np.logical_and(median, edge_img))
            edge_img[zeroed] = 0

            count = count + 1
            if count > 70:
                break
            last_median = median
            median = cv2.medianBlur(edge_img, 3)

    @staticmethod
    def find_significant_contour(edge_img):
        contours, hierarchy = cv2.findContours(
            edge_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Find level 1 contours
        level_1_meta = []
        for contourIndex, tupl in enumerate(hierarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl.copy(), 0, [contourIndex])
                level_1_meta.append(tupl)

        # From among them, find the contours with large surface area.
        contours_with_area = []
        for tupl in level_1_meta:
            contourIndex = tupl[0]
            contour = contours[contourIndex]
            area = cv2.contourArea(contour)
            contours_with_area.append([contour, area, contourIndex])

        contours_with_area.sort(key=lambda meta: meta[1], reverse=True)
        largest_contour = contours_with_area[0][0]

        return largest_contour

    def remove_background(self, image_path, cutout_path):
        src = cv2.imread(image_path, 1)
        blurred = cv2.GaussianBlur(src, (5, 5), 0)

        blurred_float = blurred.astype(np.float32) / 255.0
        edges = self.edgeDetector.detectEdges(blurred_float) * 255.0
        # cv2.imwrite('edge-raw.jpg', edges)

        edges_8u = np.asarray(edges, np.uint8)
        self.filter_out_salt_pepper_noise(edges_8u)
        # cv2.imwrite('edge.jpg', edges_8u)

        contour = self.find_significant_contour(edges_8u)
        # Draw the contour on the original image
        contour_img = np.copy(src)
        cv2.drawContours(contour_img, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
        # cv2.imwrite('contour.jpg', contour_img)

        mask = np.zeros_like(edges_8u)
        cv2.fillPoly(mask, [contour], 255)

        # calculate sure foreground area by dilating the mask
        mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

        # mark inital mask as "probably background"
        # and mapFg as sure foreground
        trimap = np.copy(mask)
        trimap[mask == 0] = cv2.GC_BGD
        trimap[mask == 255] = cv2.GC_PR_BGD
        trimap[mapFg == 255] = cv2.GC_FGD

        # visualize trimap
        trimap_print = np.copy(trimap)
        trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
        trimap_print[trimap_print == cv2.GC_FGD] = 255

        # cv2.imwrite('trimap.png', trimap_print)

        # run grabcut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
        cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        # create mask again
        mask2 = np.where(
            (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
            255,
            0
        ).astype('uint8')

        # cv2.imwrite('mask2.jpg', mask2)

        contour2 = self.find_significant_contour(mask2)
        mask3 = np.zeros_like(mask2)
        cv2.fillPoly(mask3, [contour2], 255)

        # blended alpha cut-out
        mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
        mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
        alpha = mask4.astype(float) * 1.1  # making blend stronger
        alpha[mask3 > 0] = 255
        alpha[alpha > 255] = 255
        alpha = alpha.astype(float)

        foreground = np.copy(src).astype(float)
        foreground[mask4 == 0] = 0
        background = np.zeros_like(foreground, dtype=float) * 255

        # cv2.imwrite('foreground.png', foreground)
        # cv2.imwrite('background.png', background)
        # cv2.imwrite('alpha.png', alpha)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha / 255.0
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        # Add the masked foreground and background.
        cutout = cv2.add(foreground, background)

        cv2.imwrite(cutout_path, cutout)
