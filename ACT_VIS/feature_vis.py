# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import cv2
caffe_root = '/home/ee/btech/ee1130454/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
#!/bin/bash

import glob
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
import matplotlib.pyplot as plt
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min() + 1)
    print data.shape
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    from scipy.misc import imsave
    imsave("weight_vis.png",data)
    #import pdb;pdb.set_trace()
   # data.plt.savefig()
    #fig = plt.figure()
    #from PIL import Image
    #img=Image.fromarray(data,'RGB')
    #img.save('po1.png')
    #fig.savefig(imgdata, format='png')
    #cv2.imwrite('po.png',data)
if os.path.isfile('/home/ee/btech/ee1130454/ACT_VIS/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_gpu()
model_def = '/home/ee/btech/ee1130458/deploy.prototxt'
model_weights = '/home/ee/btech/ee1130454/ACT_VIS/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
#net.blobs['data'].reshape(200,        # batch size
                       #   3,         # 3-channel (BGR) images
                        #  227, 227)  # image size is 227x227
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1) 
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
image = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_image


### perform classification
net.forward()
filters = net.params['conv1'][0].data
print "Trump was here"
#import pdb;pdb.set_trace()
vis_square(filters.transpose(0, 2, 3, 1))

# feat = net.blobs['conv1'].data[0, :36]
# vis_square(feat)

