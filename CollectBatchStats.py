import matplotlib.pylab as plt
import tensorflow as tf


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        # self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        # self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
