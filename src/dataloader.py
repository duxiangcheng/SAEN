from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from os import walk
from os.path import join
import random
import sys
import six
import lmdb
from torchvision.transforms import Compose, ToTensor, Resize


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

class devdata(Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.gtRoot = gtRoot
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize)
    
    def __getitem__(self, index):
        path = self.imageFiles[index].split('/')[-1]
        img = Image.open(self.imageFiles[index])
        gt = Image.open(join(self.gtRoot, path))
        inputImage = self.ImgTrans(img.convert('RGB'))

        groundTruth = self.ImgTrans(gt.convert('RGB'))
        
        return inputImage, groundTruth, path
    
    def __len__(self):
        return len(self.imageFiles)

class ErasingData(Dataset):
    def __init__(self, dataRoot, loadSize, training=True):
        super(ErasingData, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize)
        self.training = training
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        # mask = Image.open(self.imageFiles[index].replace('all_images','mask'))
        mask = Image.open(self.imageFiles[index].replace('all_images','Stroke'))
        gt = Image.open(self.imageFiles[index].replace('all_images','all_labels'))
        # import pdb;pdb.set_trace()
        if self.training:
        # ### for data augmentation
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)   
            all_input = random_rotate(all_input)
            img = all_input[0]
            mask = all_input[1]
            gt = all_input[2]
        ### for data augmentation
        inputImage = self.ImgTrans(img.convert('RGB'))
        mask = self.ImgTrans(mask.convert('RGB'))
        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]
       # import pdb;pdb.set_trace()

        return inputImage, groundTruth, mask, path
    
    def __len__(self):
        return len(self.imageFiles)

class LmdbDataset(Dataset):
    def __init__(self, dataRoot, loadSize, training=True):
        super(LmdbDataset, self).__init__()
        self.env = lmdb.open(dataRoot, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (dataRoot))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize)
        self.training = training
    
    def __getitem__(self, index):
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            gtbuf = txn.get(label_key)
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            mask_key = 'stroke-%09d'.encode() % index
            maskbuf = txn.get(mask_key)
            imgnameKey = 'imgname-%09d'.encode() % index
            imgname = txn.get(imgnameKey).decode('utf-8')

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            bufgt = six.BytesIO()
            bufgt.write(gtbuf)
            bufgt.seek(0)

            bufmask = six.BytesIO()
            bufmask.write(maskbuf)
            bufmask.seek(0)

            img = Image.open(buf)
            mask = Image.open(bufmask)
            gt = Image.open(bufgt)

            if self.training:
            ### for data augmentation
                all_input = [img, mask, gt]
                all_input = random_horizontal_flip(all_input)   
                all_input = random_rotate(all_input)
                img = all_input[0]
                mask = all_input[1]
                gt = all_input[2]

            inputImage = self.ImgTrans(img.convert('RGB'))
            mask = self.ImgTrans(mask.convert('RGB'))
            groundTruth = self.ImgTrans(gt.convert('RGB'))
            path = imgname+".jpg"

            return inputImage, groundTruth, mask, path
    
    def __len__(self):
        return self.nSamples