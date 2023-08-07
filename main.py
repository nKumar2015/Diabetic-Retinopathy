#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 11 16:19:09 2023

@author: nakulkumar
"""
import os
import pathlib
import numpy as np 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
np.random.seed(10)

archive = "./images/"
data_dir = pathlib.Path(archive).with_suffix('')

batch_size = 128
img_height = 224
img_width = 224

train_portion = 0.8
val_portion = 0.1

train, train_labels = [], []
val, val_labels = [], []
test, test_labels = [], []

for eye_class in os.listdir('./images/'):
  class_dir = './images/'+eye_class+'/'

  images = [class_dir+fname for fname in os.listdir(class_dir)]
  np.random.shuffle(images)

  train_idx = int(len(images)*train_portion)
  val_idx = int(len(images)*val_portion)

  train.extend(images[0:train_idx])
  val.extend(images[train_idx: val_idx + train_idx])
  test.extend(images[val_idx + train_idx:])

  train_labels.extend([eye_class]*(train_idx))
  val_labels.extend([eye_class]*(val_idx))
  test_labels.extend([eye_class]*(len(images) - train_idx - val_idx))

ohe = OneHotEncoder(categories = 'auto')
train_ohe = ohe.fit_transform(np.array(train_labels).reshape(-1,1)).toarray()
val_ohe = ohe.fit_transform(np.array(val_labels).reshape(-1,1)).toarray()
test_ohe = ohe.fit_transform(np.array(test_labels).reshape(-1,1)).toarray()

class EyeImageGen(tf.keras.utils.Sequence):
    
    def __init__(self, X, Y, Y_ohe, dir='./images/'):
        self.x = X
        self.y = Y
        self.y_ohe = Y_ohe
        self.batch_size = batch_size
    
    def on_epoch_end(self):
        return
    
    def __getitem__(self, index):
        x_path = self.x[index*batch_size: ((index+1)*batch_size)]
        X = []
      
        for x in x_path:
            im = Image.open(x).convert("RGB")
            X.append(np.asarray(im))
        
        return np.array(X), np.array(self.y_ohe[index*batch_size: ((index+1)*batch_size)])
    
    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
ilepath = './Models/Resnet50Exp0.epoch{epoch:02d}-val_accuracy{val_accuracy:.2f}-training_accuracy{accuracy:.2f}-val_loss{val_loss:.2f}-training_loss{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_weights_only=True, 
                             mode='max',
                             save_freq='epoch')

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None, 
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=5,
)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    EyeImageGen(train, train_labels, train_ohe),
    validation_data=EyeImageGen(val, val_labels, val_ohe),
    epochs=5,
    batch_size=batch_size,
    callbacks=[checkpoint]
)


#model = keras.models.load_model('./models/model.epoch01-accuracy0.75.hdf5')
#for i, path in enumerate(test):
#    im = Image.open(path).convert("RGB")
#    test[i] = np.asarray(im)
#test = np.asarray(test)

#pred = np.rint(model.predict(test))
#conf_matrix = confusion_matrix(test_ohe.argmax(axis=1), pred.argmax(axis=1))
#disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['mild', 'moderate', 'none', 'proliferate', 'servere',])
#disp.plot()
#plt.savefig('./results/resnet50.png')

    
