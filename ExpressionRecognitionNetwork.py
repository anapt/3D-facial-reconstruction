from FaceNet3D import FaceNet3D as Helpers
import matplotlib.pyplot as plt
from LoadDataset import LoadDataset
import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TF_FORCE_GPU_ALLOW_GROWTH = True
tf.compat.v1.enable_eager_execution()
print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
print("\n\n\n\n")


class ExpressionRecognition(Helpers):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # self.emotions = {
        #     "anger": 0,
        #     "disgust": 1,
        #     "fear": 2,
        #     "happiness": 3,
        #     "neutral": 4,
        #     "sadness": 5,
        #     "surprise": 6
        # }
        self.emotions = {
            "happiness": 0,
            "neutral": 1,
            "sadness": 2,
            "surprise": 3
        }

        self.em = list(self.emotions.keys())
        self.em.sort()

        self.WEIGHT_DECAY = 0.001
        # self.WEIGHT_DECAY = 0.000001
        self.BASE_LEARNING_RATE = 0.01

        self.BATCH_SIZE = 8
        self.BATCH_ITERATIONS = 400

        self.SHUFFLE_BUFFER_SIZE = 400

        self.checkpoint_dir = "./DATASET/training/expression/"
        self.checkpoint_path = "./DATASET/training/expression/cp-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor='loss',
                                                              verbose=0, save_best_only=True,
                                                              save_weights_only=True, mode='min', save_freq='epoch')

        self.history_list = list()
        self.model = self.build_model()

        self.latest = self.checkpoint_dir + "cp-0024.ckpt"

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[self.expression_dim, ]),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(len(self.em), activation=tf.nn.softmax)
        ])

        return model

    def compile(self):
        """
        Compiles the Keras model. Includes metrics to differentiate between the two main loss terms
        """
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print('Model Compiled!')

    def training(self):

        self.compile()
        model = self.model

        keras_ds = LoadDataset().load_data_for_expression()
        keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(
            self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)

        history_1 = model.fit(keras_ds, epochs=24, steps_per_epoch=steps_per_epoch,
                              callbacks=[self.cp_callback])

        self.history_list.append(history_1)

    def load_model(self):
        """
        Load trained model and compile

        :return: Compiled Keras model
        """
        self.build_model()
        self.model.load_weights(self.latest)

        self.compile()

    def model_predict(self, vector_path):
        """
        Predict out of image_path
        :param vector_path: path
        :return:
        """
        vector = np.loadtxt(vector_path)
        vector = tf.transpose(tf.constant(vector))
        # vector = np.transpose(vector)
        vector = tf.reshape(vector, shape=[1, self.expression_dim])
        x = self.model.predict(vector)

        return x

    def plots(self):
        for i in range(0, len(self.history_list)):

            plt.figure()
            plt.title('Mean Squared Error, phase %d' % i)
            plt.plot(self.history_list[i].history['accuracy'])
            plt.savefig(self.plot_path + 'mse%d.pdf' % i)

            plt.figure()
            plt.title('Mean Absolute Error, phase %d' % i)
            plt.plot(self.history_list[i].history['loss'])
            plt.savefig(self.plot_path + 'mae%d.pdf' % i)

    # def evaluate_model(self):
    #     """
    #     Evaluate model on validation data
    #     """
    #     test_ds = LoadDataset().load_dataset_single_image(self._case)
    #     loss, mse, mae = self.model.evaluate(test_ds)
    #     print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
    #           "Mean Absolute Error: {2}\n".format(loss, mse, mae))


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7 * 1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    train = ExpressionRecognition()
    # print("\n")
    # print("Batch size: %d" % train.BATCH_SIZE)
    #
    # print("\nPhase 1\nSTART")
    #
    # # with tf.device('/device:CPU:0'):
    # train.compile()

    # train.training()
    #
    # print("Phase 1: COMPLETE")
    # # print("\n \n \nPhase 2\n START")
    # #
    # # # train.training_phase_2()
    # #
    # # print("Phase 2: COMPLETE")
    # print("Saving plots...")
    #
    # train.plots()
    train.load_model()

    x = train.model_predict("./DATASET/expression/sadness/test_{:06}.txt".format(0))
    print(x*100)

main()
