#!/usr/bin/env sh
#export PYTHONPATH=$PYTHONPATH:/home/cmcc/caffe-master/examples/mtcnn-caffe/12net
export PYTHONPATH=$PYTHONPATH:~/FY/mtcnn-caffe/12net

TOOLS=~/libraries/caffe/build/tools
set -e

$TOOLS/caffe train \
	 --solver=./solver.prototxt \
