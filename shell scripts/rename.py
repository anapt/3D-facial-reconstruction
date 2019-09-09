import pathlib
import os

emotions = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

em = list(emotions.keys())
em.sort()
print(em)
data_root = '/home/anapt/expression_recognition/anger/'

for key in em:
    # print(key)
    root = "/home/anapt/expression_recognition/%s/" % key
    data_root = pathlib.Path(root)

    all_image_paths = list(data_root.glob('*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # print(all_image_paths)
    # all_image_paths = all_image_paths[0:10]
    os.chdir(root)
    # os.system("ls")
    print(len(all_image_paths))
    # for i, path in enumerate(all_image_paths):
    #     new_name = "{:06}.jpg".format(i)
    #
    #     # print(new_name)
    #     os.system("mv " + path + " ./" + new_name)
