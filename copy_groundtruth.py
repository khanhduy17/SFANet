import os
import glob
import numpy as np
import h5py
import shutil

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

src_dir = '/mmlabworkspace/WorkSpaces/KhanhND/CSP/content/drive/MyDrive/Detect_Face_Wearing_Mask_Project/Extracted_Frames'
#src_dir = '/mmlabworkspace/WorkSpaces/KhanhND/CSR_Data_Full_NoMask/ground_truth_test'
des_dir = '/mmlabworkspace/WorkSpaces/KhanhND/SFANet-crowd-counting/NFM/test_data/images'

train_file_list = './test_list_rambalac.txt'

with open(train_file_list) as f:
    train_list = f.readlines()
train_list = [l.strip('\n\r') for l in train_list]

for filename in train_list:
    src_gt_path = os.path.join(src_dir, filename)#.replace('.jpg','.mat')
    des_gt_path = os.path.join(des_dir, filename)#.replace('.jpg','.mat')
    ensure_dir(des_gt_path)
    if os.path.exists(src_gt_path):
        shutil.copyfile(src_gt_path, des_gt_path)
    else:
        print(src_gt_path)
        break
            



