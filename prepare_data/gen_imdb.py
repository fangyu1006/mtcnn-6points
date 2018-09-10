import numpy as np
import numpy.random as npr
import sys
import cv2
import os
import cPickle as pickle
import argparse

def view_bar(num, total):
	rate = float(num) / total
	rate_num = int(rate * 100)+1
	r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
	sys.stdout.write(r)
	sys.stdout.flush()


def gen_imdb(net):
	if net == "PNet":
		size = 12
	elif net == "RNet":
		size = 24
	elif net == "ONet":
		size = 48

	with open('imglists/%s/cls_%s.txt'%(net,net), 'r') as f:
		clss = f.readlines()
	with open('imglists/%s/roi_%s.txt'%(net,net), 'r') as f:
		roii = f.readlines()
	with open('imglists/%s/pts_%s.txt'%(net,net), 'r') as f:
		ptss = f.readlines()

	# generate cls.imdb
	
	print("start create cls.imdb")
	cls_list = []
	cur_ = 0
	sum_ = len(clss)
	print(sum_)
	for line in clss:
		view_bar(cur_,sum_)
		cur_ += 1

		words = line.split()
		image_file_name = words[0]
		im = cv2.imread(image_file_name)
		if im is None:
			print(image_file_name)
			continue
	 #	print(image_file_name)
		h,w,ch = im.shape
		if h!=size or w!=size:
			im = cv2.resize(im,(size,size))
		im = np.swapaxes(im, 0, 2)
		im = (im - 127.5)/127.5
		label = words[1]
		roi = [-1,-1,-1,-1]
		pts	= [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1.-1.-1]
		if len(words) > 2:
			roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]

		cls_list.append([im,label,roi,pts])

	fid = open("imglists/%s/cls.imdb"%net,'w')
	pickle.dump(cls_list, fid)
	fid.close()
	print "\n",str(len(cls_list))," Classify Data have been read into Memory..."



	# generate roi.imdb
	print("start create roi.imdb")
	roi_list = []
	cur_ = 0
	sum_ = len(roii)
	print(sum_)
	for line in roii:
		view_bar(cur_,sum_)
		cur_ += 1

		words = line.split()
		image_file_name = words[0]
		im = cv2.imread(image_file_name)
		h,w,ch = im.shape
		if h!=size or w!=size:
			im = cv2.resize(im,(size,size))
		im = np.swapaxes(im, 0, 2)
		im = (im - 127.5)/127.5
		label = words[1]
		roi = [-1,-1,-1,-1]
		pts	= [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1.-1.-1]
		roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]
		roi_list.append([im,label,roi,pts])

	fid = open("imglists/%s/roi.imdb"%net,'w')
	pickle.dump(roi_list, fid)
	fid.close()
	print "\n",str(len(roi_list))," Regression Data have been read into Memory..."


	# generate pts.imdb
	print("start create pts.imdb")
	pts_list = []
	cur_ = 0
	sum_ = len(ptss)
	print(sum_)
	for line in ptss:
		view_bar(cur_,sum_)
		cur_ += 1

		words = line.split()
		image_file_name = words[0]
		im = cv2.imread(image_file_name)
		h,w,ch = im.shape
		if h!=size or w!=size:
			im = cv2.resize(im,(size,size))
		im = np.swapaxes(im, 0, 2)
		im = (im - 127.5)/127.5
		label = words[1]
		roi = [-1,-1,-1,-1]
		pts	= [float(words[2]), float(words[3]), float(words[4]), float(words[5]), float(words[6]), float(words[7]), float(words[8]), float(words[9]), float(words[10]), float(words[11]), float(words[12]), float(words[13])]
		pts_list.append([im,label,roi,pts])

	fid = open("imglists/%s/pts.imdb"%net,'w')
	pickle.dump(pts_list, fid)
	fid.close()
	print "\n",str(len(pts_list))," Points Data have been read into Memory..."



def parse_args():
    parser = argparse.ArgumentParser(description='Generate image list',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net', dest='net', help='Net type, can be PNet, RNet, or ONet', 
                        default='PNet', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print 'Called with argument:'
    print args 
    gen_imdb(args.net)
