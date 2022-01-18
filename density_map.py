import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_density(img_path, gt):
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', gt).replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gt = mat["image_info"][0, 0][0, 0][0]
    count = 0
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][1]) >= 0 and int(gt[i][0]) < img.shape[1] and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    print('Ignore {} wrong annotation.'.format(len(gt) - count))
    k = gaussian_filter(k, 5)
    return k, count

root = '/mmlabworkspace/WorkSpaces/KhanhND/SFANet-crowd-counting/NFM'

NFM_train = os.path.join(root, 'test_data', 'images')

train_file_list = './test_list_rambalac.txt'

with open(train_file_list) as f:
    img_paths = f.readlines()
img_paths = [l.strip('\n\r') for l in img_paths]

for img_path in img_paths:
    img_path = os.path.join(NFM_train, img_path)
    print(img_path)
    
    k, count = compute_density(img_path, 'ground_truth')
    att = k > 0.001
    att = att.astype(np.float32)
    
    k_mask, count_mask = compute_density(img_path, 'ground_truth_mask')
    k_nomask, count_nomask = compute_density(img_path, 'ground_truth_nomask')
    
    ensure_dir(img_path.replace('.jpg', '.h5').replace('images', 'new_data'))
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'new_data'), 'w') as hf:
        hf['density_mask'] = k_mask
        hf['gt_mask'] = count_mask
        hf['density_nomask'] = k_nomask
        hf['gt_nomask'] = count_nomask
        hf['attention'] = att
        
