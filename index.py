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

# list_label = ['Healthy', 'Brain_Tumor']
# list_folder = ['train', 'test', 'val']

# for folder in list_folder:
#     for label in list_label:
#         print(f'jumlah {label} di data {folder} :',
#             len(os.listdir(f'F:\\4.SEKOLAH\\2. SIC\\New folder\\Brain Tumor Data Set\\{folder}\\{label}')))

path_normal = "F:\\4.SEKOLAH\\2. SIC\\New folder\\Brain Tumor Data Set\\train\\Healthy"
path_tumor = "F:\\4.SEKOLAH\\2. SIC\\New folder\\Brain Tumor Data Set\\train\\Brain_Tumor"

def display_image(images_path, label, num_images=5):
    image_file = os.listdir(images_path)
    random_images = random.sample(image_file, min(num_images, len(image_file)))

    plt.figure(figsize=(15, 5))
    for i, image_file in enumerate(random_images):
        image_path = os.path.join(images_path, image_file)
        img = plt.imread(image_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')
    
    plt.show()

display_image(path_normal, 'Normal')
display_image(path_tumor, 'Tumor')