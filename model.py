import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import glob
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

dataset = "F:\\4.SEKOLAH\\2. SIC\\Brain Tumor Classification Model\\Brain Tumor Data Set"
list_label = ['Healthy', 'Brain_Tumor']
list_folder = ['train', 'test', 'val']

for folder in list_folder:
    for label in list_label:
        label_path = os.path.join(dataset, folder, label)

image_path = []
labels = []

for folder in list_folder:
    for label in list_label:
        label_path = os.path.join(dataset, folder, label)
        if os.path.exists(label_path):
            for img in os.listdir(label_path):
                image_path.append(os.path.join(label_path, img))
                labels.append(label)

img_df = pd.DataFrame({'images': image_path, 'labels': labels})

if not img_df.empty:
    train_df, test_df = train_test_split(img_df, test_size=0.2,stratify=img_df['labels'], random_state=42)
else:
    print("data kosong")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='images',
    y_col='labels',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='images',
    y_col='labels',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Input(shape=(224, 224, 3)))

# convulational layer 1
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# layer 2
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))

# layer 3
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# model.summary()

# train model
# history_cnn = model.fit(train_generator, validation_data=test_generator, epochs=10)

model.save('model.h5')