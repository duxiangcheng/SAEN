import fire
import os
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, gtPath, maskPath, outputPath, checkValid=True):
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    imgs = [i.split('.')[0] for i in sorted(os.listdir(inputPath)) if '.jpg' in i]
    nSamples = len(imgs)
    for index in imgs:
        imagePath = os.path.join(inputPath, index+'.jpg')
        labelPath = os.path.join(gtPath, index+'.jpg')
        strokePath = os.path.join(maskPath, index+'.jpg')
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        with open(labelPath, 'rb') as f:
            labelBin = f.read()
        with open(strokePath, 'rb') as f:
            maskBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', index)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue
        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        strokeKey = 'stroke-%09d'.encode() % cnt
        imgnameKey = 'imgname-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = labelBin
        cache[strokeKey] = maskBin
        cache[imgnameKey] = index.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)