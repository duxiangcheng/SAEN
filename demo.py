import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from src.model import SGNet

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for inference')
parser.add_argument('--imgPath', type=str, default='')
parser.add_argument('--loadSize', type=int, default=512, help='image loading size')
parser.add_argument('--savedPath', type=str, default='result.jpg')
args = parser.parse_args()

cuda = torch.cuda.is_available()
netG = SGNet(3)
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True
    netG = netG.cuda()

netG.load_state_dict(torch.load(args.pretrained))
for param in netG.parameters():
    param.requires_grad = False
img_ori = cv2.imread(args.imgPath) / 255
image = cv2.resize(img_ori, (args.loadSize, args.loadSize))
image = image.transpose(2, 0, 1)
img = torch.from_numpy(image).float().unsqueeze(0)
img = img.cuda()
x_o1, x_o2, x_o3, g_image, mm = netG(img, masks=None, training=False)
ndarr = g_image.data.squeeze().float().clamp_(0, 1).cpu().numpy()
if ndarr.ndim == 3:
    img = np.transpose(ndarr, (1, 2, 0))
ndarr = np.uint8((img*255.0).round())
cv2.imwrite(args.savedPath, ndarr)