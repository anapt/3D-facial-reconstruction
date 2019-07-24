import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
from caffe import layers as L
from caffe import params as P
import tempfile
import os

caffe_root = '/home/anapt/Documents/caffe/'
sys.path.insert(0, caffe_root + 'python')

# set display defaults
# large imagesi
plt.rcParams['figure.figsize'] = (10, 10)
# don't interpolate: show square pixels
plt.rcParams['image.interpolation'] = 'nearest'
# use grayscale output rather than a (potentially misleading) color heatmap
plt.rcParams['image.cmap'] = 'gray'



caffe.set_mode_cpu()

class InverseFaceNetModel(object):

    def __init__(self):
        # Parameters
        self.weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

        self.IMG_SIZE = 240
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        self.WEIGHT_DECAY = 0.001
        self.BASE_LEARNING_RATE = 0.01

        self.BATCH_SIZE = 20
        self.BATCH_ITERATIONS = 75000

        self.SHUFFLE_BUFFER_SIZE = 1000

        # Parameters for Loss
        self.PATH = './DATASET/model2017-1_bfm_nomouth.h5'
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.photo_loss = 0
        self.reg_loss = 0

        # Model
        # self.model = self.caffenet()
        # Print a model summary
        # self.model.summary()

        self.weight_param = dict(lr_mult=1, decay_mult=1)
        self.bias_param = dict(lr_mult=2, decay_mult=0)
        self.learned_param = [self.weight_param, self.bias_param]

        self.frozen_param = [dict(lr_mult=0)] * 2

        # Loss Function
        # self.loss_func = self.model_loss()

    def conv_relu(self, bottom, ks, nout, stride=1, pad=0, group=1,
                  weight_filler=dict(type='gaussian', std=0.01),
                  bias_filler=dict(type='constant', value=0.1)):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                             num_output=nout, pad=pad, group=group,
                             param=self.learned_param, weight_filler=weight_filler,
                             bias_filler=bias_filler)
        return conv, L.ReLU(conv, in_place=True)

    def fc_relu(self, bottom, nout,
                weight_filler=dict(type='gaussian', std=0.005),
                bias_filler=dict(type='constant', value=0.1)):
        fc = L.InnerProduct(bottom, num_output=nout, param=self.learned_param,
                            weight_filler=weight_filler,
                            bias_filler=bias_filler)
        return fc, L.ReLU(fc, in_place=True)

    @staticmethod
    def max_pool(bottom, ks, stride=1):
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

    def caffenet(self, data, train=True, num_outputs=257, learn_all=False):



        # def caffenet(data, label=None, train=True, num_classes=1000,
        #              classifier_name='fc8', learn_all=False):
        """Returns a NetSpec specifying CaffeNet, following the original proto text
           specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
        n = caffe.NetSpec()
        n.data = data
        param = self.learned_param if learn_all else self.frozen_param
        n.conv1, n.relu1 = self.conv_relu(n.data, 11, 96, stride=4)
        n.pool1 = self.max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2, n.relu2 = self.conv_relu(n.norm1, 5, 256, pad=2, group=2)
        n.pool2 = self.max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
        n.conv3, n.relu3 = self.conv_relu(n.norm2, 3, 384, pad=1)
        n.conv4, n.relu4 = self.conv_relu(n.relu3, 3, 384, pad=1, group=2)
        n.conv5, n.relu5 = self.conv_relu(n.relu4, 3, 256, pad=1, group=2)
        n.pool5 = self.max_pool(n.relu5, 3, stride=2)
        n.fc6, n.relu6 = self.fc_relu(n.pool5, 4096)
        if train:
            n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
        else:
            fc7input = n.relu6
        n.fc7, n.relu7 = self.fc_relu(fc7input, 4096)
        if train:
            n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
        else:
            fc8input = n.relu7
        # always learn fc8 (param=learned_param)
        fc8 = L.InnerProduct(fc8input, num_output=num_outputs, param=self.learned_param)
        # give fc8 the name specified by argument `classifier_name`
        # n.__setattr__(classifier_name, fc8)
        if not train:
            n.probs = L.Softmax(fc8)
        # if vector is not None:
        #     n.vector = vector
        #     n.loss = L.SoftmaxWithLoss(fc8, n.label)
        #     n.acc = L.Accuracy(fc8, n.label)

        # write the net to a temporary file and return its filename
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with open(tmp.name, "w") as f:
            f.write(str(n.to_proto()))
            return f.name

    def inverse_face_net(self, train=True, learn_all=False, subset=None):
        if subset is None:
            subset = 'train' if train else 'test'
        data_root = '/home/anapt/PycharmProjects/thesis/DATASET/images/'
        sem_root = '/home/anapt/PycharmProjects/thesis/DATASET/semantic/'
        # source = caffe_root + 'data/flickr_style/%s.txt' % subset
        # transform_param = dict(mirror=train, crop_size=227,
        #                        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
        image_data = L.ImageData(
            source=data_root,
            batch_size=self.BATCH_SIZE)
        print(image_data)
        return self.caffenet(data=image_data, train=train,
                             learn_all=learn_all)

def main():
    net = InverseFaceNetModel()
    # dummy_data = L.DummyData(shape=dict(dim=[net.BATCH_SIZE, 3, net.IMG_SIZE, net.IMG_SIZE]))
    # imagenet_net_filename = net.caffenet(data=dummy_data, train=False)
    # imagenet_net = caffe.Net(imagenet_net_filename, net.weights, caffe.TEST)
    # print(imagenet_net)
    untrained_style_net = caffe.Net(net.inverse_face_net(train=False, subset='train'),
                                    net.weights, caffe.TEST)
    untrained_style_net.forward()
    style_data_batch = untrained_style_net.blobs['data'].data.copy()
    style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)

main()