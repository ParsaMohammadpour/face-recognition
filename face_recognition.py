#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import numpy as np


# In[2]:


def encode_image(img, known_face_locations=None):
    if known_face_locations == None:
        return fr.face_encodings(img)[0]
    return fr.face_encodings(img, known_face_locations=known_face_locations)[0]


# In[3]:


def is_same_face(img1_encoded, img2_encoded):
    return fr.compare_faces([img1_encoded], img2_encoded)[0]


# In[4]:


PATH1 = 'train\Aaron_Peirsol\Aaron_Peirsol_0001.jpg'
# PATH2 = 'train\Aaron_Peirsol\Aaron_Peirsol_0002.jpg'
# PATH1 = 'train\Queen_Elizabeth_II\Queen_Elizabeth_II_0004.jpg'
PATH2 = 'train\Queen_Elizabeth_II\Queen_Elizabeth_II_0009.jpg'

# different
img1 = cv2.imread(PATH1)
enc1 = encode_image(img1)
img2 = cv2.imread(PATH2)
enc2 = encode_image(img2)

print('same.' if is_same_face(enc1,enc2) else 'different.')


# In[5]:


PATH1 = 'train\Aaron_Peirsol\Aaron_Peirsol_0001.jpg'
PATH2 = 'train\Aaron_Peirsol\Aaron_Peirsol_0002.jpg'
# PATH1 = 'train\Queen_Elizabeth_II\Queen_Elizabeth_II_0004.jpg'
# PATH2 = 'train\Queen_Elizabeth_II\Queen_Elizabeth_II_0009.jpg'

# different
img1 = cv2.imread(PATH1)
enc1 = encode_image(img1)
img2 = cv2.imread(PATH2)
enc2 = encode_image(img2)

print('same.' if is_same_face(enc1,enc2) else 'different.')


# In[ ]:


def encode_person(path, name):
    image_path = path + '/' + name + "/*"
    first_image_path = glob.glob(image_path)[0]
    img = cv2.imread(first_image_path)
    result = None
    try:
        img = cv2.imread(first_image_path)
        result = encode_image(img)
    except:
        print(name)
        result = encode_image(img, [(75, 70, 190, 180)])
    return result

def make_df(path):
    people = [f.name for f in os.scandir(path) if f.is_dir()]
    df = pd.DataFrame({'name':people})
    df['encode'] = df.apply(lambda x: encode_person(path, x['name']), axis=1)
    df.to_csv('face_recognition_df.csv')
    return df

TRAIN_PATH = 'train'
df = make_df(TRAIN_PATH)
df


# In[ ]:


def load_df(path):
    return pd.read_csv(path)

def make_array(string):
    return np.array([float(i) for i in string[1:-1].split()])

DF_PATH = 'face_recognition_df.csv'
df = load_df(DF_PATH)
df['encode'] = df['encode'].apply(lambda x: make_array(x))
df


# In[ ]:


def test(path):
    names = [f.name for f in os.scandir(path) if f.is_dir()]
    for name in names:
        image_path = path + '/' + name + '/*'
        first_image_path = glob.glob(image_path)[0]
        img = cv2.imread(first_image_path)
        encode = None
        try:
            encode = encode_image(img)
        except:
            encode = encode_image(img, [(75, 70, 190, 180)])
        
        has_match = False
        for n, e in df[['name', 'encode']].values:
            if is_same_face(encode, e) and n == name:
                print('matched.')
                print('name: ' + n)
                print('---------------------------------------------------')
                has_match = True
                break
        if not has_match:
            print('no match.')
            print(name)
            print('***************************************************')

test('test')

