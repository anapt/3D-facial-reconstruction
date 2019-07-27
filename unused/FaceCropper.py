import cv2


class FaceCropper(object):
    CASCADE_PATH = "./DATASET/haarcascade_frontalface_default.xml"

    def __init__(self):
        """
        Class initializer
        """
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, img, save_image, n):
        """
        Detect face and crop to desired dimensions
        :param img: input image containing a single face in black background
        :param save_image: boolean if True save image
        :param n: number of iteration, used when save_image is True
        :return: type <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if faces is None:
            print('Failed to detect face')
            return 0
        elif len(faces) == 1:
            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny + nr, nx:nx + nr]
                lastimg = cv2.resize(faceimg, (240, 240))

            if save_image:
                cropped_image_path = ("./DATASET/images/image_%d.png" % n)
                cv2.imwrite(cropped_image_path, lastimg)

            return lastimg
