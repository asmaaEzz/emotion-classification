from flask import Flask, render_template, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import tensorflow as tf


app = Flask(__name__)

model = tf.keras.Sequential([

    tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(3, activation='softmax')
])
model.load_weights('models/32-0.825.hdf5')

def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, -1)
    img = img / 255
    return img


@app.route('/')

def upload_file():

    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_files():

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print(f)
        test_image = read_image(f.filename)
        pred = model.predict_classes(np.array([test_image]))
        res= str(pred)
        print(pred)
        if res == '[1]':
            return render_template('nutral.html')
        elif res == '[0]':
            return render_template('negative.html')
        else:
            return render_template('positive.html')


if __name__ == '__main__':
    app.debug = True
    app.run()