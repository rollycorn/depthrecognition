# preprocessing of input images
#   method: normalizatoin

import os
import stat
import skimage
import sys
import shutil
from skimage import io
import numpy as np
import math

outdir = 'preprocessed'
testoutdir = 'test_preprocessed'
test_ = False

# true if file has one of extention in exts
def hasExt( file, exts ):
  for ext in exts:
   if file.find(ext) + len(ext) == len(file):
     return True
  return False

# true if filepath is image
def isImg( filepath ):
 return os.stat( filepath ).st_mode != stat.S_ISDIR and \
   hasExt( filepath, ['jpg', 'jpeg', 'png', 'gif', 'tif', 'png', 'exr'] )

# preprocess images in dir
def preprocess( dir ):
  files = os.listdir( dir )
  for file in files:
    filepath = dir + '/' + file
    if isImg( filepath ):
      img = io.imread( filepath )
      # extract rgb and convert into float32
      img = img.swapaxes( 0, 2 )
      img = np.array([img[0],img[1],img[2]], dtype=np.float32).swapaxes( 0, 2 )
      # normalize
      ave = np.average(img)
      img -= ave
      sigma = math.sqrt( np.average(img*img) )
      img /= sigma
      # save
      savepath = outdir + '/' + file + '.exr'
      io.imsave( savepath, img, plugin='freeimage' )
      # cancel 180deg rotation of imsave
      img2 = io.imread( savepath, plugin='freeimage' )
      io.imsave( savepath, img2, plugin='freeimage' )
      # for test
      if test_:
        for x in range( 0, len(img) ):
          for y in range( 0, len(img[0]) ):
            for c in range( 0, len(img[0][0]) ):
              col = img[x][y][c]
              col = col * 100 + 128
              if col < 0:
                col = 0
              if col > 255:
                col = 255
              img[x][y][c] = col
        img = np.array( img, dtype=np.uint8 )
        io.imsave( testoutdir + '/' + file, img )

# dump image
def dumpImg( dir ):
  files = os.listdir( dir )
  for file in files:
    filepath = dir + '/' + file
    print filepath
    img = io.imread( filepath, plugin='freeimage' )
    print img

# preprocess images in directory for deep learning
# SYNOPSIS
#   preprocess input_directory [test|dump]
# DESCRIPTION
#   output directory : 'preprocessed'
#   test output(png) directory : 'test_preprocessed'
#   These directories are removed first.
if __name__ == '__main__':
  args = sys.argv
  if len( args ) >= 3 and args[2] == 'test':
    test_ = True
  if len( args ) >= 3 and args[2] == 'dump':
    dumpImg( args[1] )
  else:
    if os.path.exists( outdir ):
      shutil.rmtree( outdir )
    os.mkdir( outdir )
    if os.path.exists( testoutdir ):
      shutil.rmtree( testoutdir )
    os.mkdir( testoutdir )
    if len( args ) >= 2:
      preprocess( args[1] )
