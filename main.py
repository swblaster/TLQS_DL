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
import feeder
import model
import os
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    
    #Load train data
    Y_train=feeder.read_train_data()
    
    #Load test data
    print("================ Load test data...================")
    test_imgs = []
    dir = "./test_image/"
    file_list = os.listdir(dir)
    image_files = [file for file in file_list]
    image_names = [os.path.splitext(file)[0] for file in file_list]
    for img_file in image_files:
        img_path = os.path.join(dir, img_file)
        try:
            img = plt.imread(img_path)
            print(f"Loaded image: {img_file}")
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
        test_imgs.append(img)

    #Define Model
    model = model.Model()
    
    # Hyperparameters for the experiment: 
    epoch_num=50
    batch_size = 32
    num_train_batches= Y_train.shape[0]//batch_size + 1
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    log_dir = "log/fit/vdsr"  
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    for epoch in range(epoch_num):
        psnr_bicubic_mean = 0
        psnr_output_mean = 0
        
        mode = 0
        if mode == 0: # our random signal
            print ("Using our random signal!")
            Y_shuffled_train=[]
            random_shuffle_list=np.load('integer.npy')
            for i in range(len(Y_train)):
                Y_shuffled_train.append(Y_train[random_shuffle_list[i]])
            Y_train_tensor = tf.data.Dataset.from_tensor_slices(Y_shuffled_train).batch(batch_size)
        elif mode == 1: # python random
            print ("Using python psuedo random!")
            Y_train_tensor = tf.data.Dataset.from_tensor_slices(Y_train).shuffle(buffer_size=len(Y_train)).batch(batch_size)
        else: # no shuffle
            print ("No shuffle!")
            Y_shuffled_train=[]
            random_shuffle_list = range(len(Y_train))
            for i in range(len(Y_train)):
                Y_shuffled_train.append(Y_train[random_shuffle_list[i]])
            Y_train_tensor = tf.data.Dataset.from_tensor_slices(Y_shuffled_train).batch(batch_size)
        
        #Train
        loss_train = 0
        bicubic_loss_train = 0
        for Y in tqdm(Y_train_tensor, desc=f'Epoch {epoch} Training', total=num_train_batches, leave=False):
            with tf.GradientTape() as tape:
                _, loss_train_batch, bicubic_loss = model.call(Y, downSample=True, training=True)
            
            loss_train += loss_train_batch.numpy()/num_train_batches
            bicubic_loss_train += bicubic_loss.numpy()/num_train_batches
            grad = tape.gradient(loss_train_batch, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
        
        #Test
        for index, image in enumerate(test_imgs):
            
            #Test scale
            r=2 
            output,_,__ = model.call([image], [r], downSample=True)  
            output = (output*255+127.5).numpy()[0]
            output = np.clip(output,0,255).astype('uint8')
            
            image_bicubic = tf.image.resize(image, [image.shape[0]//r, image.shape[1]//r], method='bicubic',antialias=True)
            image_bicubic = tf.image.resize(image_bicubic, [image.shape[0], image.shape[1]], method='bicubic').numpy()
            image_bicubic = np.clip(image_bicubic, 0, 255).astype('uint8')
        
            psnr_bicubic = tf.image.psnr(image, image_bicubic, max_val=255)
            psnr_output= tf.image.psnr(image, output, max_val=255)
        
            psnr_bicubic_mean += psnr_bicubic.numpy()/len(test_imgs)
            psnr_output_mean += psnr_output.numpy()/len(test_imgs)
            
            #Save output images
            output_path = './output_images/epoch '+str(epoch)+'/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cv2.imwrite(output_path + f'{image_names[index]}_output.png', cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            cv2.imwrite(output_path + f'{image_names[index]}_bicubic.png', cv2.cvtColor(image_bicubic, cv2.COLOR_BGR2RGB))
            cv2.imwrite(output_path + f'{image_names[index]}_label.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(f'{image_names[index]} psnr_bicubic :{psnr_bicubic.numpy()}  psnr_output : {psnr_output.numpy()}')
        
        #TensorBoard    
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss_train, step=epoch)
            tf.summary.scalar('psnr_output_mean', psnr_output_mean, step=epoch)

        print(f"Epoch: {epoch}  learning_rate: {optimizer.learning_rate.numpy()}  Bicubic_loss: {bicubic_loss_train}  train loss: {loss_train} ")
        print(f"psnr_bicubic_mean: {psnr_bicubic_mean}  psnr_output_mean: {psnr_output_mean}  diff: {psnr_output_mean - psnr_bicubic_mean}")

        #Save checkpoint
        root = tf.train.Checkpoint(optimizer=optimizer,model=model)
        checkpoint_dir = './checkpoint/ckpt_'+ str(epoch)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        root.save(checkpoint_prefix)
        print("----------------------------------------------Checkpoint saved-------------------------------------------------")
