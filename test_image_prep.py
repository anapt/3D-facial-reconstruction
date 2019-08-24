from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
import cv2
import numpy as np
import pathlib


path = "/home/anapt/Documents/MUG/unpacked/"

data_root = pathlib.Path(path)

all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
print(len(all_image_paths))
all_image_paths = np.random.choice(all_image_paths, 5000)

for i, path in enumerate(all_image_paths):
    new_name = "{:06}.jpg".format(i)
    img = cv2.imread(path, 1)

    img = FaceCropper().generate(img, save_image=False, n=None)
    if img is None:
        continue
    img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)

    cropped_image_path = ("/home/anapt/Documents/MUG/cropped/{:06}.png".format(i))
    cv2.imwrite(cropped_image_path, img)
