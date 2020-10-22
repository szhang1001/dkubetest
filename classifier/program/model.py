from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from dkube import dkubeLoggerHook as logger_hook

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
import gc
import copy

import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam

if os.getenv('DKUBE_JOB_CLASS',None) == 'notebook':
    print("hello")
FLAGS = None
TF_TRAIN_STEPS = int(os.getenv('STEPS',1000))

if os.getenv('DKUBE_JOB_CLASS',None) == 'notebook':
    MODEL_DIR = "model"
    DATA_DIR = "/opt/dkube/input"
    LOG_DIR = "/opt/dkube/logs"
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists("/opt/dkube/logs"):
        os.makedirs("/opt/dkube/logs")
else:
    MODEL_DIR = "/opt/dkube/output"
    DATA_DIR = "/opt/dkube/input"
    LOG_DIR = "/opt/dkube/logs"
    if not os.path.exists("/opt/dkube/logs"):
        os.makedirs("/opt/dkube/logs")

import pickle
with open(DATA_DIR+"Xy_small_n100.p", "rb") as f:
    X_train,y_train = pickle.load(f) 
    
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(include_top = False, 
                               weights = None)
# base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in model.layers[:249]:
#     layer.trainable = False
# for layer in model.layers[249:]:
#     layer.trainable = True
for layer in model.layers:
    layer.trainable = True
    
def train_model_big(model,data,label,batch_size,epoch,lr,model_save_path,log_dir):
    """Train model
    Args: 
        model: a keras model
        data: image data array
        label: label array (without one-hot encoder)
        batch_size
        lr: learning rate
    Return:
        a keras History object
    """
    
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    from keras.utils.np_utils import to_categorical
    label = to_categorical(label, num_classes=2)
    index = int(data.shape[0] * 0.8)
    
    train_X = data[0:index]
    train_y = label[0:index]
    
    valid_X = data[index:]
    valid_y = label[index:]
    
#     flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')
    train_gen = ImageDataGenerator(rotation_range=30, 
                         width_shift_range= 10.0, 
                         height_shift_range= 10.0, 
                         rescale=1/255)
    valid_gen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_gen.flow(train_X, 
                     train_y, 
                     batch_size=batch_size, 
                     shuffle=True)
    valid_generator = valid_gen.flow(valid_X, valid_y, batch_size = batch_size)
    
    
#     keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    tf_record = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_check = keras.callbacks.ModelCheckpoint(filepath = model_save_path, 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, mode='auto', period=1)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

    his = model.fit_generator(train_generator, 
                        steps_per_epoch = train_X.shape[0] // batch_size, 
                        epochs=epoch,
                        validation_data=valid_generator, 
                        validation_steps= valid_X.shape[0] // batch_size, 
                        callbacks = [tf_record, model_check, reduce_lr])
    
    return model,his

model_save_path = MODEL_DIR + '/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
model, his= train_model_big(model,X_train,y_train,40,10,0.1,
                            model_save_path = model_save_path,
                            log_dir = LOG_DIR)

model.save(MODEL_DIR+'/model0.hdf5')