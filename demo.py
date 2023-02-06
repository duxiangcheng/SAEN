import argparse
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image
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
netG.eval()
img = Image.open(args.imgPath)
transform = Compose([
        Resize(size=args.loadSize, interpolation=Image.BICUBIC),
        ToTensor(),])
img_PIL_Tensor = transform(img).unsqueeze(0)
img = img_PIL_Tensor.cuda()
x_o1, x_o2, x_o3, g_image, mm = netG(img, masks=None, training=False)
g_image = g_image.data.cpu()
save_image(g_image, args.savedPath)