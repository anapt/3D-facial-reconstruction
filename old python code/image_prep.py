import LandmarkDetection as ld
import cv2
import FaceCropper as fc

image_path = "./DATASET/4.jpg"

cut = ld.LandmarkDetection()

image = cv2.imread(image_path)

image = cut.crop_image(image)

out = cut.cutout_mask_array(image, None, flip_rgb=False, save_image=False)
cv2.imwrite("./DATASET/face_db/3.png", out)

cv2.imshow("", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # image = crop.generate(image, False, None)
# cv2.imshow("", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
