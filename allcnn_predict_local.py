#!/usr/bin/env python
# coding: utf-8

# In[5]:


from __future__ import print_function
import tensorflow
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import h5py
from urllib.request import urlopen


# import logging
# logger = tensorflow.get_logger()
# logger.setLevel(logging.ERROR)

model_json= "allcnn_model.json"
model_weights= "allcnn_weights.hdf5"
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def loadModel(json_desc, weights):

    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
               
    loaded_model.load_weights(model_weights)
    return loaded_model
    
model = loadModel(model_json, model_weights)

def transformImage(image):    
    data = np.reshape(img, (1, 32, 32, 3))
    data = data.astype('float32')
    data /= 255
    return data
    
def predictImage(image):
    imgarr = np.asarray(image)
    imaget = transformImage(imgarr)
    result = model.predict(imaget)
    result = result[0].tolist()
    best_index=result.index(max(result))
    best_prob = result[best_index]
    
    result[best_index] = 0    
    sec_best_index=result.index(max(result))
    sec_best_prob = result[sec_best_index]

    result[sec_best_index] = 0    
    thd_best_index=result.index(max(result))
    thd_best_prob = result[thd_best_index]
    
    return best_prob, best_index, sec_best_prob, sec_best_index, thd_best_prob, thd_best_index
    
def processImage(url: str):
    img = Image.open(urlopen(url))
    img = img.resize((32, 32))   
    return img;

url = "https://jndfiles-pub.s3.eu-west-1.amazonaws.com/images/dogs/dogs-8.jpg"
img = processImage(url)
best_prob, best_index, sec_best_prob, sec_best_index, thd_best_prob, thd_best_index = predictImage(url)
print ("1st prediction : "+str(categories[best_index])+" prob : "+'{:05.3f}'.format(best_prob))
print ("2nd prediction : "+str(categories[sec_best_index])+" prob : "+'{:05.3f}'.format(sec_best_prob))
print ("3nd prediction : "+str(categories[thd_best_index])+" prob : "+'{:05.3f}'.format(thd_best_prob))



