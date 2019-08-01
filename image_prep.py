import LandmarkDetection as ld
import cv2
import FaceCropper as fc

image_path = "./DATASET/images/val/val.jpg"

cut = ld.LandmarkDetection()
crop = fc.FaceCropper()

image = cv2.imread(image_path)



# cv2.imshow("", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


out = cut.cutout_mask_array(image, None, flip_rgb=False, save_image=False)
cv2.imwrite("landmark_cut.png", out)

# image = crop.generate(image, False, None)
cv2.imshow("", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
