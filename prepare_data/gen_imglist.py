import numpy as np
import numpy.random as npr
import os
import argparse

net = "PNet"

def gen_imaglist(data_dir, net):

    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48

    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(data_dir,'%s/landmark_%s_aug.txt' %(size,size)), 'r') as f:
        landmark = f.readlines()
    
    dir_path = os.path.join(data_dir, 'imglists')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
        os.makedirs(os.path.join(dir_path, "%s" %(net)))

    # pos and neg for face detection
    with open(os.path.join(dir_path, "%s" %(net),"cls_%s.txt" % (net)), "w") as f:
        print("Generating cls data...")
        if net == "PNet":    
            nums = [len(neg), len(pos), len(part)]
            ratio = [3, 1, 1]
            #base_num = min(nums)
            base_num = 250000
            print(len(neg), len(pos), len(part), base_num)
            if len(neg) > base_num * 3:
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            pos_keep = npr.choice(len(pos), size=base_num, replace=True)
            part_keep = npr.choice(len(part), size=base_num, replace=True)

            for i in pos_keep:
                f.write(pos[i])
            for i in neg_keep:
                f.write(neg[i])

        else:
            for i in np.arange(len(pos)):
                f.write(pos[i])
            for i in np.arange(len(neg)):
                f.write(neg[i])
        print len(neg)
        print len(pos)


    # pos and part for box regression
    with open(os.path.join(dir_path, "%s" %(net),"roi_%s.txt" % (net)), "w") as f:
        print("Generating roi data...")
        if net == "PNet":    
            nums = [len(neg), len(pos), len(part)]
            ratio = [3, 1, 1]
            #base_num = min(nums)
            base_num = 250000
            print(len(neg), len(pos), len(part), base_num)
            if len(neg) > base_num * 3:
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            pos_keep = npr.choice(len(pos), size=base_num, replace=True)
            part_keep = npr.choice(len(part), size=base_num, replace=True)

            for i in pos_keep:
                f.write(pos[i])
            for i in part_keep:
                f.write(part[i])

        else:
            for i in np.arange(len(pos)):
                f.write(pos[i])
            for i in np.arange(len(part)):
                f.write(part[i])
        print len(pos)
        print len(part)

    # landmarks for landmark regression
    with open(os.path.join(dir_path, "%s" %(net),"pts_%s.txt" % (net)), "w") as f:
        print("Generating pts data...")
        if net == "PNet":    
            nums = [len(neg), len(pos), len(part)]
            ratio = [3, 1, 1]
            #base_num = min(nums)
            base_num = 250000
            print(len(neg), len(pos), len(part), base_num)
            if len(neg) > base_num * 3:
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            pos_keep = npr.choice(len(pos), size=base_num, replace=True)
            part_keep = npr.choice(len(part), size=base_num, replace=True)
            for item in landmark:
                f.write(item)

        else:

            for i in np.arange(len(landmark)):
                f.write(landmark[i])
        print len(landmark)



def parse_args():
    parser = argparse.ArgumentParser(description='Generate image list',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', help='data directory',
                        default='.', type=str)
    parser.add_argument('--net', dest='net', help='Net type, can be PNet, RNet, or ONet', 
                        default='PNet', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print 'Called with argument:'
    print args 
    gen_imaglist(args.data_dir, args.net)


