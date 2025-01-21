
import glob
import shutil

import cv2
import numpy as np 
from sklearn.model_selection import train_test_split


def generate_sketch(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    invert_img = cv2.bitwise_not(gray_img)                    #invert_img=255-grey_img
    blur_img = cv2.GaussianBlur(invert_img, (111,111), 0)
    invblur_img = cv2.bitwise_not(blur_img)                   #invblur_img=255-blur_img
    sketch_img = cv2.divide(gray_img, invblur_img, scale=256.0)
    stacked_img = np.stack((sketch_img)*3, axis=-1)

img_list = sorted(glob.glob("../Dataset/photos/*.jpg"))

# split the dataset into train, test and validation
train_images, test_images = train_test_split(img_list, test_size = 0.2, random_state = 0)
test_images, val_images = train_test_split(test_images, test_size = 0.05, random_state = 0)
print(len(train_images), len(test_images), len(val_images))

train_images_path = "CelebAHQ_synthesized/photo/train/"
test_images_path = "CelebAHQ_synthesized/photo/test/"
val_images_path = "CelebAHQ_synthesized/photo/val/"
train_sketches_path = "CelebAHQ_synthesized/sketch/train/"
test_sketches_path = "CelebAHQ_synthesized/sketch/test/"
val_sketches_path = "CelebAHQ_synthesized/sketch/val/"

for img_path in train_images:
    img = cv2.imread(img_path)
    sketch = generate_sketch(img)
    # concat = cv2.hconcat([img, stacked_img])
    img_name = img_path.split("/")[-1]
    sketch_img_name = img_name
    shutil.copy(img_path, train_images_path+img_name)
    cv2.imwrite(train_sketches_path+sketch_img_name, sketch)

for img_path in test_images:
    img = cv2.imread(img_path)
    sketch = generate_sketch(img)
    # concat = cv2.hconcat([img, stacked_img])
    img_name = img_path.split("/")[-1]
    sketch_img_name = img_name
    shutil.copy(img_path, test_images_path+img_name)
    cv2.imwrite(test_sketches_path+sketch_img_name, sketch)

for img_path in val_images:
    img = cv2.imread(img_path)
    sketch = generate_sketch(img)
    # concat = cv2.hconcat([img, stacked_img])
    img_name = img_path.split("/")[-1]
    sketch_img_name = img_name
    shutil.copy(img_path, val_images_path+img_name)
    cv2.imwrite(val_sketches_path+sketch_img_name, sketch)

print("Dataset Created...!!")