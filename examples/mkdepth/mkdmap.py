#!/usr/bin/env python
import sys
args=sys.argv
sys.path.append('../../python')
sys.path.append('../../python/caffe')

import caffe
import skimage
from skimage import io
import os
import shutil
import sys

net = None
outlayer = ''

def processSingleImage( input_path, output_path ):
  # in_img : (H, W, K)
  in_img = io.imread( input_path, plugin='freeimage' )
  in_img_ = in_img.reshape( 1, in_img.shape[2], in_img.shape[1], in_img.shape[0] )
  out = net.forward(**{net.inputs[0]:in_img_})
  out_img = out[outlayer]
  # print out_img
  out_img_ = out_img[0][0]
  io.imsave( output_path, out_img_, plugin='freeimage' )
  # cancel 180deg rotation of imsave
  tmp_img = io.imread( output_path, plugin='freeimage' )
  io.imsave( output_path, tmp_img, plugin='freeimage' )


if __name__ == '__main__':
  if len(args) < 6:
    print "make depth map from single image"
    print "arg1 : input image directory"
    print "arg2 : output image directory"
    print "arg3 : learned model"
    print "arg4 : prototxt"
    print "arg5 : outlayer"
    sys.exit()

  outlayer = args[5]
  caffe.set_mode_cpu()
  net = caffe.Net(args[4], 1, weights=args[3])

  in_dir = args[1]
  out_dir = args[2]
  files = os.listdir( in_dir )
  for file in files:
    input_path = in_dir + '/' + file
    output_path = out_dir + '/' + file
    processSingleImage( input_path, output_path )
