import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *

from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
import tensorflow as tf
from keras import regularizers


#def dice_coef(y_true, y_pred, smooth=1):
    
  #  y_true = keras.print_tensor(y_true, message='y_true = ')
   # y_pred = keras.print_tensor(y_pred, message='y_pred = ')
 #   y_true = keras.flatten(y_true)
   # y_pred = keras.flatten(y_pred)  
  #  intersection = keras.sum(y_true * y_pred)
    #union = keras.sum(y_true) + keras.sum(y_pred)
    #return keras.mean( (2. * intersection + smooth) / (union + smooth))

def dice_loss(y_true, y_pred):
    
    smooth=1
  #  y_true = keras.print_tensor(y_true, message='y_true = ')
   # y_pred = keras.print_tensor(y_pred, message='y_pred = ')
    y_true_f=keras.flatten(y_true)
    y_pred_f=keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    union = keras.sum(y_true_f) + keras.sum(y_pred_f)
    score= (2. * intersection + smooth) / (union + smooth)
    return 1.-score


#def dice_coef_loss(y_true, y_pred):
  #  y_true = keras.print_tensor(y_true, message='y_true = ')
  #  y_pred = keras.print_tensor(y_pred, message='y_pred = ')

    #return -dice_coef(y_true, y_pred)

#def weighted_loss(y_true, y_pred):
 #   def weighted_binary_cross_entropy(y_true, y_pred):
        

  #      w = tf.reduce_sum(y_true)/tf.cast(tf.size(y_true), tf.float32)
        #real_th = 0.5-th 
        #tf_th = tf.fill(tf.shape(y_pred), real_th) 
        #tf_zeros = tf.fill(tf.shape(y_pred), 0.)
        
   #     return (1.0 - w) * y_true * - tf.log(tf.sigmoid(y_pred) ) + (1- y_true) * w * -tf.log(1 - tf.sigmoid(y_pred))
   

    #return weighted_binary_cross_entropy(y_true,y_pred)

#def loss2(y_true, y_pred):
 #   def dice_loss(y_true, y_pred):
  #      numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
   #     denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    #    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    #return keras.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def weighted_cross_entropy(beta=3):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss2(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss2 = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss2)

  return loss2





def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)


    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
   
    model = Model(inputs = inputs, outputs= conv10)

    model.compile(optimizer = Adam(lr = 8e-5), loss = dice_loss, metrics = ["accuracy"])
  #  model.compile(optimizer = SGD(lr=0.01, nesterov=True), loss = dice_loss, metrics = ['accuracy'])
    
   
#    #model.compile(optimizer = Adam(lr = 1e-4), loss =iou_coef_loss, metrics = ['accuracy'])
    #model.summary() 'binary_crossentropy'

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model