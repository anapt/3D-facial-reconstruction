import ImagePreprocess as image_preprocess
import time


def main():
    """
    Script used to create the synthetic dataset
    """
    # Number of images to create
    N = 20000
    preprocess = image_preprocess.ImagePreprocess()

    for n in range(20000, 50000):
        start = time.time()
        preprocess.create_image_and_save(n)
        # print("Time passed:", time.time() - start)
        print(n)


main()
