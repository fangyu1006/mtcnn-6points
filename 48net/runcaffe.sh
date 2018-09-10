#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:~/FY/mtcnn-caffe/48net

set -e
~/libraries/caffe/build/tools/caffe train \
	 --solver=./solver.prototxt \
  	 #--weights=./48net.caffemodel
