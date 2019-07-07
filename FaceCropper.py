import cv2


class FaceCropper(object):
    CASCADE_PATH = "./DATASET/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, cropped_image_path, show_result, save_image):
        img = cv2.imread(image_path)
        if img is None:
            print("Can't open image file")
            return 0

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if faces is None:
            print('Failed to detect face')
            return 0
        elif len(faces) == 1:

            if show_result:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_image:
                for (x, y, w, h) in faces:
                    r = max(w, h) / 2
                    centerx = x + w / 2
                    centery = y + h / 2
                    nx = int(centerx - r)
                    ny = int(centery - r)
                    nr = int(r * 2)

                    faceimg = img[ny:ny+nr, nx:nx+nr]
                    lastimg = cv2.resize(faceimg, (240, 240))
                    # i += 1
                    cv2.imwrite(cropped_image_path, lastimg)
