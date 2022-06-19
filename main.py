import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.models import load_model

model = load_model("model-devolepment\pa_rice.h5")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #print('upload_image filename: ' + filename)

        def predict():
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = image.load_img(
                path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            if classes[0][0] == 1:
                result = "Arborio"

            elif classes[0][1] == 1:
                result = 'Basmati'
            elif classes[0][2] == 1:
                result = 'Ipsala'
            elif classes[0][3] == 1:
                result = 'Jasmine'
            elif classes[0][4] == 1:
                result = 'Karacadag'
            else:
                result = 'Tidak Terdeteksi'
            return result

        flash(predict())
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
