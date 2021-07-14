from model import *
from data import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import *
import os

data_gen_args = dict(rotation_range=180,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.0,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_gen = trainGenerator(batch_size=8,train_path='data/train',image_folder='image',label_folder='label',aug_dict=data_gen_args,save_to_dir =None)


valid_gen = valGenerator(batch_size=4,val_path="data/val",image_folder='image',label_folder='label')


model = unet()

model_checkpoint = ModelCheckpoint('lane_segment_model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
early_stopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
red_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5, verbose=0, mode='auto',min_lr=8e-7)

#steps_per_epoch should typically be equal to the number of samples of your 
#dataset divided by the batch size. Here 50/8~7

#Validation_steps should typically be equal to the number of samples of your val 
#dataset divided by the batch size. Here 20/4=5


mod=model.fit(train_gen,validation_data=valid_gen,validation_steps=5,steps_per_epoch=7,epochs=10,callbacks=[model_checkpoint,early_stopping,red_lr])

