import pathlib
import os

root = "./DATASET/bootstrapping/predict/"
data_root = pathlib.Path(root)

all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths.sort()

# all_image_paths = all_image_paths[20000:25000]

print(len(all_image_paths))
for i, path in enumerate(all_image_paths):
    new_name = "im_{:06}.png".format(i)

    os.system("mv " + path + " ./DATASET/bootstrapping/predict/" + new_name)
