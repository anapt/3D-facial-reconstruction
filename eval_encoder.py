from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from LoadDataset import LoadDataset
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from FaceNet3D import FaceNet3D as Helpers
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TF_FORCE_GPU_ALLOW_GROWTH = False
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
print("\n\n\n\n")


class ModelEvaluate(Helpers):
    def __init__(self):
        super().__init__()
        self.latest = self.trained_models_dir + "cp-phase1.ckpt"
        self._case = 'validation'
        self.net = InverseFaceNetEncoderPredict()
        self.encoder = InverseFaceNetEncoder()
        self.model = self.load_model()

    def load_model(self):
        """
        Load trained model and compile

        :return: Compiled Keras model
        """
        self.encoder.build_model()
        model = self.encoder.model
        model.load_weights(self.latest)

        self.encoder.compile()
        # model = self.encoder.model

        return model

    def call_evaluation(self):
        self.model.load_weights(self.latest)
        test_ds = LoadDataset().load_dataset_single_image(self._case)
        loss, mse, mae = self.model.evaluate(test_ds)
        print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
              "Mean Absolute Error: {2}\n".format(loss, mse, mae))


def main():
    eval = ModelEvaluate()
    eval.call_evaluation()


main()