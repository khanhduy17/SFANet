import torch
from torch.utils import data
from dataset import Dataset
from models import Model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SHA', type=str, help='dataset')
parser.add_argument('--data_path', default=r'D:\dataset', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'D:\checkpoint\SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])

model.eval()
with torch.no_grad():
    mae_mask, mse_mask = 0.0, 0.0
    mae_nomask, mse_nomask = 0.0, 0.0
    for i, (images, gt_mask, gt_nomask) in enumerate(test_loader):
        image_path = test_dataset.__getname__(i)
        print(image_path)
        
        images = images.to(device)

        predict_mask, predict_nomask, _ = model(images)
        
        save_name = image_path.replace('/images', '/output').replace('.jpg', '.txt')
        ensure_dir(save_name)
        with open(save_name, "w") as file:
            face_num1 = str(predict_mask.sum().item()) + "\n"
            face_num2 = str(predict_nomask.sum().item()) + "\n"
            file.write(face_num1)
            file.write(face_num2)

        print('predict_mask:{:.2f} label_mask:{:.2f}'.format(predict_mask.sum().item(), gt_mask.item()))
        print('predict_nomask:{:.2f} label_nomask:{:.2f}'.format(predict_nomask.sum().item(), gt_nomask.item()))
        mae_mask += torch.abs(predict_mask.sum() - gt_mask).item()
        mse_mask += ((predict_mask.sum() - gt_mask) ** 2).item()
        mae_nomask += torch.abs(predict_nomask.sum() - gt_nomask).item()
        mse_nomask += ((predict_nomask.sum() - gt_nomask) ** 2).item()

    mae_mask /= len(test_loader)
    mse_mask /= len(test_loader)
    mse_mask = mse_mask ** 0.5
    print('MAE_mask:', mae_mask, 'MSE_mask:', mse_mask)
    
    mae_nomask /= len(test_loader)
    mse_nomask /= len(test_loader)
    mse_nomask = mse_nomask ** 0.5
    print('MAE_nomask:', mae_nomask, 'MSE_nomask:', mse_nomask)
