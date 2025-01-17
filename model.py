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
import feeder

class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        self.channel_num=3
        super(Model, self).__init__(**kwargs)
        self.augmentation = tf.keras.Sequential([
          layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
          layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        self.cnn_in = tf.keras.layers.Conv2D(input_shape=(None, None, self.channel_num), kernel_size=3, 
                                        filters=64, padding='same', activation='relu', kernel_initializer='he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(0.0001))
        self.cnns = tf.keras.Sequential()
        for i in range(18):
            self.cnns.add(tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same',activation='relu', 
                                      kernel_initializer='he_normal', kernel_regularizer = tf.keras.regularizers.l2(0.0001)))
        self.cnn_out = tf.keras.layers.Conv2D(kernel_size=3, filters=self.channel_num, padding='same', activation='tanh',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.0001))
        
    def call(self, imgs, r=[2,3,4], downSample=False, training=False):
        if training==True:
            
            #Data augmentation
            l = [imgs]
            for i in range(3):
                l.append(self.augmentation(imgs))
            l = [tf.cast(img, tf.uint8) for img in l]
            imgs = tf.stack(l, axis=0)
            imgs = tf.reshape(imgs, [-1,imgs.shape[2], imgs.shape[3], self.channel_num])
            
        #Normalization
        imgs = tf.cast(imgs, dtype=tf.float32)
        imgs = imgs - 127.5
        imgs = imgs/255
        
        #down_sampling
        r_rand = random.choice(r)
        H = imgs.shape[1]
        W = imgs.shape[2]
            
        if downSample==False:
            X = tf.image.resize(imgs, [H*r_rand,W*r_rand], method='bicubic')
        if downSample==True:
            #Make ILR images
            r_rand = random.choice(r)
            H = imgs.shape[1]
            W = imgs.shape[2]
            H_reduced = int(H/r_rand)
            W_reduced = int(W/r_rand)
            
            X = tf.image.resize(imgs, [H_reduced,W_reduced], method='bicubic',antialias=True)
            
            X = tf.image.resize(X, [H,W], method='bicubic')
            
            
        #Residual learning
        residual = self.cnn_in(X)
        residual = self.cnns(residual)
        residual = self.cnn_out(residual)
        output = tf.keras.layers.Add()((residual, X))#Skip connection
        
        if downSample==False:
            return output
        loss = tf.reduce_mean(tf.square(imgs-output))
        bicubic_loss = tf.reduce_mean(tf.square(imgs-X))
        
        return output, loss, bicubic_loss
        
