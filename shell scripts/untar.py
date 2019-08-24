import pathlib
import os


data_root = '/home/anapt/Documents/MUG/subjects3/'

data_root = pathlib.Path(data_root)

all_image_paths = list(data_root.glob('**/*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]

print(all_image_paths)
os.chdir("/home/anapt/Documents/MUG/unpacked/")
# # os.system("ls")

for i, path in enumerate(all_image_paths):
    new_name = "{:06}.jpg".format(i)
    # print(new_name)
    os.system("cp " + path + ' ./' + new_name)

