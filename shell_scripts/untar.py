import pathlib
import os
from PIL import Image

# data_root = '/home/anapt/Documents/MUG/subjects3/'
#
# data_root = pathlib.Path(data_root)
#
# all_image_paths = list(data_root.glob('**/*.jpg'))
# all_image_paths = [str(path) for path in all_image_paths]
#
# print(all_image_paths)
# os.chdir("/home/anapt/Documents/MUG/unpacked/")
# # # os.system("ls")
#
# for i, path in enumerate(all_image_paths):
#     new_name = "{:06}.jpg".format(i)
#     # print(new_name)
#     os.system("cp " + path + ' ./' + new_name)

data_root = '/home/anapt/Documents/colorferet/colorferet/dvd1/data/images/'

data_root = pathlib.Path(data_root)

all_image_paths = list(data_root.glob('**/*fb.ppm'))
all_image_paths = [str(path) for path in all_image_paths]

# print(all_image_paths)
# all_image_paths = all_image_paths[0:10]
# os.chdir("/home/anapt/Documents/MUG/unpacked/")
# # os.system("ls")

# for i, path in enumerate(all_image_paths):
#     new_name = "{:06}.ppm".format(i)
#
#     print(new_name)
#     os.system("sudo bzip2 -d " + path)


def is_grey_scale(img):
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b: return False
    return True


for i, path in enumerate(all_image_paths):
    new_name = "{:06}.png".format(2523+i)
    im = Image.open(path)
    if is_grey_scale(im):
        continue
    print(new_name)
    im.save("/home/anapt/Documents/dataset2/" + new_name)
