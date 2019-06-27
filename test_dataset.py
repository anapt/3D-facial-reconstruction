import cv2
import numpy as np

import face_cropper as fc
import background_remover as br

n = 0
image_path = ("./DATASET/images/im_%d.png" % n)
cutout_path = ("./DATASET/images/cutout/im_%d.png" % n)
cropped_image_path = ("./DATASET/images/cropped/image_%d.png" % n)

remover = br.BackgroundRemover()
remover.remove_background(image_path, cutout_path)

detector = fc.FaceCropper()
detector.generate(cutout_path, cropped_image_path, False, True)
