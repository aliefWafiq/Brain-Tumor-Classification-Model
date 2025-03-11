import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

#img_path = 'F:\\4.SEKOLAH\\2. SIC\\Brain Tumor Classification Model\\Brain Tumor Data Set\\val\\Healthy\\Not Cancer  (1475).jpg'
img_path = 'F:\\4.SEKOLAH\\2. SIC\\Brain Tumor Classification Model\\Brain Tumor Data Set\\val\\Brain_tumor\\Cancer (1531).jpg'
img = image.load_img(img_path, target_size=(244, 244))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

model = tf.keras.models.load_model('model.h5')

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("The image is not a brain tumor.")
    print(prediction[0][0])
else:
    print(prediction[0][0])
    print("The image is a brain tumor.")