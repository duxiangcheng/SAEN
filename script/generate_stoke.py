import glob2
import os
import cv2
import numpy as np
threshold = 35
input_dir = 'SCUT-EnsText/train_sets'
save_str = "stroke"

if not os.path.exists(os.path.join(input_dir, save_str)):
    os.mkdir(os.path.join(input_dir, save_str))
images = glob2.glob(os.path.join(input_dir, 'all_images', '*.jpg'))
print("#item: ", len(images))

for i, item in enumerate(images):
    label_img_path = os.path.join(input_dir, 'all_labels', os.path.basename(item))
    img = cv2.imread(item)
    label_img = cv2.imread(label_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

    diff = np.abs(gray_img.astype(np.float32) - gray_label_img.astype(np.float32))
    diff_threshold = threshold
    diff[diff < diff_threshold] = 1
    diff[diff >= diff_threshold] = 0
    diff = (diff * 255).astype("uint8")
    cv2.imwrite(os.path.join(input_dir, save_str, os.path.basename(item)), diff)