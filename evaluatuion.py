import math
import argparse
import numpy as np
from torch.utils.data import DataLoader
from src.dataloader import devdata
from scipy import signal, ndimage


parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='',
                    help='results')
parser.add_argument('--gt_path', type=str, default='',
                    help='labels')
args = parser.parse_args()

sum_psnr = 0
sum_ssim = 0
sum_AGE = 0 
sum_pCEPS = 0
sum_pEPS = 0
sum_mse = 0

count = 0
sum_time = 0.0
l1_loss = 0

img_path = args.target_path
gt_path = args.gt_path

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
  #  import pdb;pdb.set_trace()
    mu1 = signal.fftconvolve(img1, window, mode = 'valid')
    mu2 = signal.fftconvolve(img2, window, mode = 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode = 'valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode = 'valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode = 'valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map = True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())

    sign_mcs = np.sign(mcs[0 : level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0 : level - 1]), weight[0 : level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power

imgData = devdata(dataRoot=img_path, gtRoot=gt_path)
data_loader = DataLoader(imgData, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

for k, (img,lbl,path) in enumerate(data_loader):
	##import pdb;pdb.set_trace()
	mse = ((lbl - img)**2).mean()
	sum_mse += mse
	print(path,count, 'mse: ', mse)
	if mse == 0:
		continue
	count += 1
	psnr = 10 * math.log10(1/mse)
	sum_psnr += psnr
	print(path,count, ' psnr: ', psnr)
    
	R = lbl[0,0,:, :]
	G = lbl[0,1,:, :]
	B = lbl[0,2,:, :]

	YGT = .299 * R + .587 * G + .114 * B

	R = img[0,0,:, :]
	G = img[0,1,:, :]
	B = img[0,2,:, :]

	YBC = .299 * R + .587 * G + .114 * B
	Diff = abs(np.array(YBC*255) - np.array(YGT*255)).round().astype(np.uint8)
	AGE = np.mean(Diff)
	print(' AGE: ', AGE) 
	mssim = msssim(np.array(YGT*255), np.array(YBC*255))
	sum_ssim += mssim
	print(count, ' ssim:', mssim)
	threshold = 20

	Errors = Diff > threshold
	EPs = sum(sum(Errors)).astype(float)
	pEPs = EPs / float(512*512)
	print(' pEPS: ' , pEPs)
	sum_pEPS += pEPs
	########################## CEPs and pCEPs ################################
	structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
	sum_AGE+=AGE
	erodedErrors = ndimage.binary_erosion(Errors, structure).astype(Errors.dtype)
	CEPs = sum(sum(erodedErrors))
	pCEPs = CEPs / float(512*512)
	print(' pCEPS: ' , pCEPs)
	sum_pCEPS += pCEPs

print(sum_psnr)
print('avg mse:', sum_mse / count)
print('average psnr:', sum_psnr / count)
print('average ssim:', sum_ssim / count)
print('average AGE:', sum_AGE / count)
print('average pEPS:', sum_pEPS / count)
print('average pCEPS:', sum_pCEPS / count)
