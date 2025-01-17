'''
Jihyun Lim <jhades625@naver.com>
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
01/18/2025
'''
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras import layers
import random
import os

def read_train_data():
    imgs = []
    sub_image_size = 41

    #Load train data
    image_dir = "./291/"
    file_list = os.listdir(image_dir)
    image_files = [file for file in file_list if file.endswith((".bmp", ".jpg"))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    print("================ Load train data... ================")
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            img = plt.imread(img_path)
            print(f"Loaded image: {img_file}")
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")

       #Cropping image
        img = tf.image.extract_patches(np.array([img]), [1, sub_image_size, sub_image_size, 1], [
                                1, sub_image_size, sub_image_size, 1], [1, 1, 1, 1], 'VALID')
        
        img = tf.reshape(img, [-1, sub_image_size, sub_image_size, 3])
        
        imgs.append(img)
        
    imgs = np.vstack(imgs)
    
    return imgs



