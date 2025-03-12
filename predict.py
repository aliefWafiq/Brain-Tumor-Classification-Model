import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    import shutil
    shutil.rmtree('uploaded / image')
    print()
except:
    pass

model = tf.keras.models.load_model('model.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded / image'

@app.route('/')
def upload_f():
    return render_template('index.html')

def finds():
    test_datagen = ImageDataGenerator(rescale = 1./255)
    vals = ['Healthy','Brain Tumor']
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory(
        test_dir, 
        target_size = (224,224),
        Color_mode = "rgb",
        shuffle = False,
        class_mode = 'categorical',
        batch_size = 1)
    
    prediction = model.predict_generator(test_generator)
    print(prediction)
    return str(vals[np.argmax(prediction)])

@app.route('/upload', methods['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.jooin(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds()
        return render_template('index.html', ss = val)

if __name__ == '__main__':
    app.run()
#img_path = 'F:\\4.SEKOLAH\\2. SIC\\Brain Tumor Classification Model\\Brain Tumor Data Set\\val\\Healthy\\Not Cancer  (1475).jpg'
# img_path = 'F:\\4.SEKOLAH\\2. SIC\\Brain Tumor Classification Model\\Brain Tumor Data Set\\val\\Brain_tumor\\Cancer (1531).jpg'
# img = image.load_img(img_path, target_size=(244, 244))

# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# prediction = model.predict(img_array)

# if prediction[0] > 0.5:
#     print("The image is not a brain tumor.")
#     print(prediction[0][0])
# else:
#     print(prediction[0][0])
#     print("The image is a brain tumor.")