import torch
from torch import nn
from torch import optim
from torch.utils import data
from dataset import Dataset
from models import Model
from tensorboardX import SummaryWriter
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=8, type=int, help='batch size')
parser.add_argument('--epoch', default=500, type=int, help='train epochs')
parser.add_argument('--dataset', default='NFM', type=str, help='dataset')
parser.add_argument('--data_path', default=r'D:\dataset', type=str, help='path to dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--load', default=True, action='store_true', help='load checkpoint')
parser.add_argument('--save_path', default=r'D:\checkpoint\SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--log_path', default='./logs', type=str, help='path to log')

args = parser.parse_args()

train_dataset = Dataset(args.data_path, args.dataset, True)
train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

writer = SummaryWriter(args.log_path)

mseloss = nn.MSELoss(reduction='sum').to(device)
bceloss = nn.BCELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)

if args.load:
    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_mae = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae_mask'] + torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae_nomask']
    start_epoch = checkpoint['epoch'] + 1
else:
    best_mae = 999999
    start_epoch = 0

for epoch in range(start_epoch, start_epoch + args.epoch):
    loss_mask_avg, loss_nomask_avg, loss_att_avg = 0.0, 0.0, 0.0

    for i, (images, density_mask, density_nomask, att) in enumerate(train_loader):
        images = images.to(device)
        density_mask = density_mask.to(device)
        density_nomask = density_nomask.to(device)
        att = att.to(device)
        outputs_mask, outputs_nomask, attention = model(images)
        print('output_mask:{:.2f} label_mask:{:.2f}'.format(outputs_mask.sum().item() / args.bs, density_mask.sum().item() / args.bs))
        print('output_nomask:{:.2f} label_nomask:{:.2f}'.format(outputs_nomask.sum().item() / args.bs, density_nomask.sum().item() / args.bs))

        loss_mask = mseloss(outputs_mask, density_mask) / args.bs
        loss_nomask = mseloss(outputs_nomask, density_nomask) / args.bs
        loss_att = bceloss(attention, att) / args.bs * 0.1
        loss_sum = loss_mask + loss_nomask + loss_att

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        loss_mask_avg += loss_mask.item()
        loss_nomask_avg += loss_nomask.item()
        loss_att_avg += loss_att.item()

        print("Epoch:{}, Step:{}, Loss:{:.4f} {:.4f} {:.4f}".format(epoch, i, loss_mask_avg / (i + 1), loss_nomask_avg / (i + 1), loss_att_avg / (i + 1)))

    writer.add_scalar('loss/train_loss_mask', loss_mask_avg / len(train_loader), epoch)
    writer.add_scalar('loss/train_loss_nomask', loss_nomask_avg / len(train_loader), epoch)
    writer.add_scalar('loss/train_loss_att', loss_att_avg / len(train_loader), epoch)

    model.eval()
    with torch.no_grad():
        mae_mask, mse_mask = 0.0, 0.0
        mae_nomask, mse_nomask = 0.0, 0.0
        for i, (images, gt_mask, gt_nomask) in enumerate(test_loader):
            images = images.to(device)

            predict_mask, predict_nomask, _ = model(images)

            print('predict_mask:{:.2f} label_mask:{:.2f}'.format(predict_mask.sum().item(), gt_mask.item()))
            print('predict_nomask:{:.2f} label_nomask:{:.2f}'.format(predict_nomask.sum().item(), gt_nomask.item()))
            mae_mask += torch.abs(predict_mask.sum() - gt_mask).item()
            mse_mask += ((predict_mask.sum() - gt_mask) ** 2).item()
            mae_nomask += torch.abs(predict_nomask.sum() - gt_nomask).item()
            mse_nomask += ((predict_nomask.sum() - gt_nomask) ** 2).item()
            if i > 50:
                break

        #mae /= len(test_loader)
        #mse /= len(test_loader)
        mae_mask /= 50
        mse_mask /= 50
        mse_mask = mse_mask ** 0.5
        mae_nomask /= 50
        mse_nomask /= 50
        mse_nomask = mse_nomask ** 0.5
        print('Epoch:', epoch, 'MAE_mask:', mae_mask, 'MSE_mask:', mse_mask)
        print('Epoch:', epoch, 'MAE_nomask:', mae_nomask, 'MSE_nomask:', mse_nomask)
        
        writer.add_scalar('eval/MAE_mask', mae_mask, epoch)
        writer.add_scalar('eval/MSE_mask', mse_mask, epoch)
        writer.add_scalar('eval/MAE_nomask', mae_nomask, epoch)
        writer.add_scalar('eval/MSE_nomask', mse_nomask, epoch)

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mae_mask': mae_mask,
                 'mse_mask': mse_mask, 'mae_nomask': mae_nomask, 'mse_nomask': mse_nomask}
        torch.save(state, os.path.join(args.save_path, 'checkpoint_latest.pth'))

        if (mae_mask+mae_nomask) < best_mae:
            best_mae = mae_mask+mae_nomask
            torch.save(state, os.path.join(args.save_path, 'checkpoint_best.pth'))
    model.train()

writer.close()
