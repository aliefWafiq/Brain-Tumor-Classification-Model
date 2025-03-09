import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

dataset = "F:\\4.SEKOLAH\\2. SIC\\ Brain Tumor Classification Model\\Brain Tumor Data Set"
list_label = ['Healthy', 'Brain_Tumor']
list_folder = ['train', 'test', 'val']

# for folder in list_folder:
#     for label in list_label:
#         label_path = os.path.join(dataset, folder, label)

image_path = []
labels = []

for folder in list_folder:
    for label in list_label:
        label_path = os.path.join(dataset, folder, label)
        if os.path.exists(label_path):
            for i in os.listdir(label_path):
                image_path.append(os.path.join(label_path, img))
                labels.append(label)

test_image = []
test_label = []
for i in range(len(image_path)):
    img = image.load_img(image_path[i], target_size(224, 224))
    img_array = image.img_to_array(img)
    test_image.append(img_array)
    test_label.append(labels[i])

test_image = np.array(test_image)
test_image = test_image.reshape((-1, 224, 224, 3))
test_label = to_categorical(test_label, num_classes=len(list_label))

model = tf.keras.models.load_model('model.h5')

model.layers[-1].activation = tf.keras.activations.softmax

model.summary()

loss, acc = model.evaluate(test_image, test_label, verbose=2)
print("restored model accuracy: {:5.2f}%", format(100 * acc))