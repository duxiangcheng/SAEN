import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from src.dataloader import LmdbDataset
from src.loss import Loss
from src.extractor import VGG16FeatureExtractor
from src.model import SGNet
from src.config import Config

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOfWorkers', type=int, default=0, help='workers for dataloader')
    parser.add_argument('--config_path', type=str, default='./config/config.yml', help='path for saving models')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    args = parser.parse_args()
    config_path = args.config_path
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    config = Config(config_path)
    config.CHECKPOINTS = args.checkpoints
    return config

def main():
    torch.set_num_threads(5)
    config = load_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    train_data = LmdbDataset(config.DATA_ROOT, config.INPUT_SIZE, training=True)
    train_data_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUMOFWORKERS, drop_last=False, pin_memory=True)
    netG = SGNet(3)

    if config.PRETRAINED != '':
        print('loaded ')
        netG.load_state_dict(torch.load(config.PRETRAINED))
    G_optimizer = optim.Adam(netG.parameters(), lr=config.GEN_LR, betas=(config.GEN_BETA1, config.GEN_BETA2))
    criterion = Loss(VGG16FeatureExtractor(), lr=config.DIS_LR, betasInit=(config.DIS_BETA1, config.DIS_BETA2))

    numOfGPUs = torch.cuda.device_count()

    if torch.cuda.is_available():
        cuda = True
        print('Cuda is available!')
        cudnn.enable = True
        cudnn.benchmark = True
        
        netG = netG.cuda()
        criterion = criterion.cuda()
        if numOfGPUs > 1:
            netG = nn.DataParallel(netG, device_ids=range(numOfGPUs))
            criterion = nn.DataParallel(criterion, device_ids=range(numOfGPUs))

    num_epochs = config.NUM_EPOCHES

    for epoch in range(1, num_epochs + 1):
        netG.train()
        for k,(imgs, gt, masks, _) in enumerate(train_data_loader):
            if cuda:
                imgs = imgs.cuda()
                gt = gt.cuda()
                masks = masks.cuda()
            netG.zero_grad()
            
            x_o1, x_o2, x_o3, fake_images, mm = netG(imgs, masks)
            G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt)
            G_loss = G_loss.sum()
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()      

            print('[{}/{}] Generator Loss of epoch{} is {}'.format(k,len(train_data_loader), epoch , G_loss.item()))
        
        if ( epoch % 10 == 0):
            if numOfGPUs > 1 :
                torch.save(netG.module.state_dict(), config.CHECKPOINTS + '/SGNet_{}.pth'.format(epoch))
            else:
                torch.save(netG.state_dict(), config.CHECKPOINTS + '/SGNet_{}.pth'.format(epoch))

if __name__ == "__main__":
    main()