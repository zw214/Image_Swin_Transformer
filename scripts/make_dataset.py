import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

def create_image_and_labels(filenames):
    feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }
    def _parse_function(example, feature_dictionary=feature_dictionary):
        parsed_example = tf.io.parse_example(example, feature_dictionary)
        return parsed_example

    negative_images = []
    negative_labels = []
    def read_data_negative(filename):
        full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        full_dataset = full_dataset.shuffle(buffer_size=31000)
        full_dataset = full_dataset.cache()
        print("Size of Training Dataset: ", len(list(full_dataset)))
        
        feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }   

        full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(full_dataset)
        step = 0
        for image_features in full_dataset:
        #print(image_features['label'].numpy())
            if image_features['label'].numpy() == 0:
                step += 1

                if step > 1500:
                    break
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224, 224))
                image=cv2.merge([image,image,image])        
                #plt.imshow(image)
                negative_images.append(image)
                negative_labels.append(0)


    benign_images = []
    benign_labels = []
    malig_images = []
    malig_labels = []
    def read_data_benign(filename):
        full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        full_dataset = full_dataset.shuffle(buffer_size=31000)
        full_dataset = full_dataset.cache()
        print("Size of Training Dataset: ", len(list(full_dataset)))
        
        feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }   

        full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(full_dataset)
        for image_features in full_dataset:
        #print(image_features['label'].numpy())
            if image_features['label'].numpy() == 1 or image_features['label'].numpy() == 2:
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224, 224))
                image=cv2.merge([image,image,image])        
                benign_images.append(image)
                benign_labels.append(1)

    def read_data_malig(filename):
        full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
        full_dataset = full_dataset.shuffle(buffer_size=31000)
        full_dataset = full_dataset.cache()
        print("Size of Training Dataset: ", len(list(full_dataset)))
        
        feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }   

        full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(full_dataset)
        for image_features in full_dataset:
        #print(image_features['label'].numpy())
            if image_features['label'].numpy() == 3 or image_features['label'].numpy() == 4:
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224, 224))
                image=cv2.merge([image,image,image])        
                malig_images.append(image)
                malig_labels.append(2)


    filenames=filenames

    for file in filenames:
        read_data_negative(file)
        read_data_benign(file)
        read_data_malig(file)

    images = negative_images + benign_images + malig_images
    labels = negative_labels + benign_labels + malig_labels
    return images, labels

def create_train_and_test_data(images, labels):

    X=np.array(images)
    y=np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

    x_train, x_test = x_train/255.0, x_test/255.0
    y_train = keras.utils.to_categorical(y_train, 3)
    y_test = keras.utils.to_categorical(y_test, 3)

    return x_train, x_test, y_train, y_test



