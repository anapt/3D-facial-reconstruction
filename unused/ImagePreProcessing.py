from unused import FaceCropper as fc
import LandmarkDetection as ld


class ImagePreProcessing(object):

    def __init__(self, n):
        self.n = n
        self.image_path = ("./DATASET/images/im_%d.png" % self.n)
        self.cutout_path = ("./DATASET/images/cutout/im_%d.png" % self.n)
        self.cropped_image_path = ("./DATASET/images/cropped/image_%d.png" % self.n)

    def detect_crop_save(self):
        cut = ld.LandmarkDetection()
        cut.cutout_mask(self.image_path, self.cutout_path)

        crop = fc.FaceCropper()
        crop.generate(self.cutout_path, self.cropped_image_path, False, True)
