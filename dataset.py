from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional


class Dataset(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset
        if dataset == 'NFM':
            dataset = os.path.join('NFM')
        elif dataset == 'SHA':
            dataset = os.path.join('ShanghaiTech', 'part_A')
        elif dataset == 'SHB':
            dataset = os.path.join('ShanghaiTech', 'part_B')

        self.image_list = glob.glob(os.path.join(data_path, dataset, 'test_data', 'images', '**/*.jpg'),  recursive=True)
        self.label_list = glob.glob(os.path.join(data_path, dataset, 'test_data', 'new_data', '**/*.h5'),  recursive=True)
        self.image_list.sort()
        self.label_list.sort()
        

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        density_mask = np.array(label['density_mask'], dtype=np.float32)
        density_nomask = np.array(label['density_nomask'], dtype=np.float32)
        attention = np.array(label['attention'], dtype=np.float32)
        gt_mask = np.array(label['gt_mask'], dtype=np.float32)
        gt_nomask = np.array(label['gt_nomask'], dtype=np.float32)
        trans = Transforms((0.8, 1.2), (400, 400), 2, (0.5, 1.5), self.dataset)
        if self.is_train:
            image, density_mask, density_nomask, attention = trans(image, density_mask, density_nomask, attention)
            return image, density_mask, density_nomask, attention
        else:   
            height, width = image.size[1], image.size[0]
            
            height = int(round(height * 0.5))
            width = int(round(width * 0.5))
            #image = image.resize((width, height), Image.BILINEAR)
            
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt_mask, gt_nomask

    def __len__(self):
        return len(self.image_list)
    
    def __getname__(self, index):
        return self.image_list[index]


if __name__ == '__main__':
    train_dataset = Dataset('/mmlabworkspace/WorkSpaces/KhanhND/SFANet-crowd-counting', 'NFM', True)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for image, label_mask, label_nomask, att in train_loader:
        print(image.size())
        print(label_mask.size())
        print(label_nomask.size())
        print(att.size())

        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.subplot(1, 4, 2)
        plt.imshow(label_mask.squeeze(), cmap='jet')
        plt.subplot(1, 4, 3)
        plt.imshow(label_nomask.squeeze(), cmap='jet')
        plt.subplot(1, 4, 4)
        plt.imshow(att.squeeze(), cmap='jet')
        plt.show()
