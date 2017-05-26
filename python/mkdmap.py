#!/usr/bin/env python
"""
make depth map from single image
  arg1 : input image directory
  arg2 : output image directory

"""

import sys
args=sys.argv
sys.path.append('../../python')
sys.path.append('../../python/caffe')

import caffe
import skimage
from skimage import io
import os
import shutil

net = None

def processSingleImage( input_path, output_path ):
  in_img0 = io.imread( 'hoge.exr', plugin='freeimage' )
  print "in_img0:"
  print in_img0
  print "---"

  # in_img : (H, W, K)
  in_img = io.imread( input_path, plugin='freeimage' )
  in_img_ = in_img.reshape( 1, in_img.shape[2], in_img.shape[1], in_img.shape[0] )
  out = net.forward(**{net.inputs[0]:in_img_})
  out_img = out['nonlinear2']
  print out_img
  out_img_ = out_img[0][0]
  io.imsave( output_path, out_img_, plugin='freeimage' )
  # cancel 180deg rotation of imsave
  tmp_img = io.imread( output_path, plugin='freeimage' )
  io.imsave( output_path, tmp_img, plugin='freeimage' )


if __name__ == '__main__':

  caffe.set_mode_cpu()
  net = caffe.Net('deploy.prototxt', 1, weights='mkdmap_iter_1000.caffemodel')

  in_dir = args[1]
  out_dir = args[2]
  files = os.listdir( in_dir )
  for file in files:
    input_path = in_dir + '/' + file
    output_path = out_dir + '/' + file
    processSingleImage( input_path, output_path )
