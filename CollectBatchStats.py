import matplotlib.pylab as plt
import tensorflow as tf


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

# steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
#
# batch_stats_callback = CollectBatchStats()
#
# history = model.fit(image_data, epochs=2,
#                     steps_per_epoch=steps_per_epoch,
#                     callbacks = [batch_stats_callback])
#
# plt.figure()
# plt.ylabel("Loss")
# plt.xlabel("Training Steps")
# plt.ylim([0,2])
# plt.plot(batch_stats_callback.batch_losses)
#
# plt.figure()
# plt.ylabel("Accuracy")
# plt.xlabel("Training Steps")
# plt.ylim([0,1])
# plt.plot(batch_stats_callback.batch_acc)
