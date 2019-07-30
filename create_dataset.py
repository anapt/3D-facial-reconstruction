import ImagePreprocess as image_preprocess


def main():
    # Number of images to create
    N = 100
    preprocess = image_preprocess.ImagePreprocess()

    for n in range(25, N+1):
        preprocess.create_image_and_save(n)
        print(n)


main()
