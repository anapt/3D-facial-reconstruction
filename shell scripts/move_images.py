import pathlib
import os

root = "./DATASET/images/training/"
data_root = pathlib.Path(root)

all_image_paths = list(data_root.glob('*.png'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths.sort()

all_image_paths = all_image_paths[20000:24000]

print(len(all_image_paths))
for i, path in enumerate(all_image_paths):
    new_name = "im_{:06}.png".format(1000+i)

    os.system("mv " + path + " ./DATASET/images/validation/" + new_name)
