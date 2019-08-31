import cv2


class FaceCropper(object):
    CASCADE_PATH = "./DATASET/haarcascade_frontalface_default.xml"

    def __init__(self):
        """
        Class initializer
        """
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, img, save_image=False, n=None):
        """
        Detect face and crop to desired dimensions
        :param img: input image containing a single face in black background
        :param save_image: boolean if True save image
        :param n: number of iteration, used when save_image is True
        :return: type <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(224, 224))
        if faces is None:
            print('Failed to detect face')
            return 0
        elif len(faces) == 1:
            for (x, y, w, h) in faces:
                # w = h = 512
                # x = 0
                # y = y - 50
                # r = max(w, h) / 2
                # print(x, y, w, h, r)
                r = 400     # for MUG

                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny + nr, nx:nx + nr]
                try:
                    lastimg = cv2.resize(faceimg, (300, 300))

                    if save_image:
                        cropped_image_path = ("/home/anapt/Documents/MUG/cropped/{:06}.png".format(n))
                        cv2.imwrite(cropped_image_path, lastimg)

                    return lastimg

                except Exception as e:
                    print(str(e))
