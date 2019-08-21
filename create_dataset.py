import ImagePreprocess as image_preprocess
import time


def main():
    # Number of images to create
    N = 10000
    preprocess = image_preprocess.ImagePreprocess()

    for n in range(0, N):
        start = time.time()
        preprocess.create_image_and_save(n)
        # print("Time passed:", time.time() - start)
        print(n)


main()
