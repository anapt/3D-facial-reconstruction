import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe

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

model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
# model_def = '/home/anapt/Downloads/3DMM_CNN/deploy_network.prototxt'
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
# model_weights = '/home/anapt/Downloads/3DMM_CNN/3dmm_cnn_resnet_101.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# # load the mean ImageNet image (as distributed with Caffe) for subtraction
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print('mean-subtracted values: BGR', mu)
#
# # create transformer for the input called 'data'
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#
# transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
# transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
#
# # set the size of the input (we can skip this if we're happy
# #  with the default; we can also change it later, e.g., for different batch sizes)
# net.blobs['data'].reshape(50,        # batch size
#                           3,         # 3-channel (BGR) images
#                           227, 227)  # image size is 227x227
#
# image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
# transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# plt.show()
#
# # copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image
#
# ### perform classification
# output = net.forward()
#
# output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
#
# print('predicted class is:', output_prob.argmax())
#
# # load ImageNet labels
# labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
#
# labels = np.loadtxt(labels_file, str, delimiter='\t')
#
# print('output label:', labels[output_prob.argmax()])
#
# # sort top five predictions from softmax output
# top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
#
# print('probabilities and labels:', output_prob[top_inds], labels[top_inds])
#
# ''' Examining intermediate output '''
# # for each layer, show the output shape
# # activation shapes, which typically have the form (batch_size, channel_dim, height, width)
# for layer_name, blob in net.blobs.items():
#     print(layer_name + '\t' + str(blob.data.shape))
#
# # Now look at the parameter shapes. The parameters are exposed as another OrderedDict, net.params.
# # We need to index the resulting values with either [0] for weights or [1] for biases.
# # The param shapes typically have the form (output_channels, input_channels, filter_height,
# # filter_width) (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).
# for layer_name, param in net.params.items():
#     print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
#
#
# def vis_square(data):
#     """Take an array of shape (n, height, width) or (n, height, width, 3)
#        and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
#
#     # normalize data for display
#     data = (data - data.min()) / (data.max() - data.min())
#
#     # force the number of filters to be square
#     n = int(np.ceil(np.sqrt(data.shape[0])))
#     padding = (((0, n ** 2 - data.shape[0]),
#                 (0, 1), (0, 1))  # add some space between filters
#                + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
#     data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
#
#     # tile the filters into an image
#     data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#     data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#
#     plt.imshow(data)
#     plt.axis('off')
#     plt.show()
#
#
# # the parameters are a list of [weights, biases]
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))
#
# feat = net.blobs['conv1'].data[0, :36]
# vis_square(feat)
#
# feat = net.blobs['pool5'].data[0]
# vis_square(feat)
#
# feat = net.blobs['fc6'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# plt.show()
#
# feat = net.blobs['prob'].data[0]
# plt.figure(figsize=(15, 3))
# plt.plot(feat.flat)
# plt.show()
