import os
import io
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)


def predict_price_textual():
    lr_drop = load('./Models/linear_regression_drop.joblib')
    rfr_drop = load('./Models/random_forest_drop.joblib')
    lr_price = load('./Models/linear_regression_price.joblib')
    rfr_price = load('./Models/random_forest_price.joblib')


@app.route('/car', methods=["POST"])
def confirm_car():
    try:
        # File Input
        file = request.files['image']
        # Checking Supported Extension
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            return jsonify({'error': 'Unsupported file format!'})
        filename = os.path.splitext(file.filename)[0]
        filepath = './Images/'+filename+'.jpg'
        file.save(filepath)
        # Accordingly Opening File
        with open(filepath, 'r') as file:
            print(type(file))
            if isinstance(file, str):
                image = load_img(filepath, target_size=(224, 224))
            elif isinstance(file, io.BytesIO) or isinstance(file, io.TextIOWrapper):
                image = Image.open(filepath)
                image = image.resize((224, 224))
            else:
                return jsonify({'error': 'Unsupported file type!'})
        # Loading Model
        model = load_model('./Models/vgg16_model_carnocar')
        class_names = ["Not Car", "Car"]
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
        os.remove(filepath)
        return {'Car': class_names[np.argmax(score)], 'Score': 100*np.max(score)}
    except:
        return {"error": "Server Issues! Please Try Again Later!"}


@app.route('/damage', methods=["POST"])
def confirm_damage():
    try:
        # File Input
        file = request.files['image']
        # Checking Supported Extension
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            return jsonify({'error': 'Unsupported file format!'})
        filename = os.path.splitext(file.filename)[0]
        filepath = './Images/'+filename+'.jpg'
        file.save(filepath)
        # Accordingly Opening File
        with open(filepath, 'r') as file:
            print(type(file))
            if isinstance(file, str):
                image = load_img(filepath, target_size=(256, 256))
            elif isinstance(file, io.BytesIO) or isinstance(file, io.TextIOWrapper):
                image = Image.open(filepath)
                image = image.resize((256, 256))
            else:
                return jsonify({'error': 'Unsupported file type!'})
        # Loading Model
        model_name = request.form['model_name']
        model = load_model('./Models/bestModels/{}'.format(model_name))
        class_names = ["damaged", "not"]
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
        os.remove(filepath)
        return {'Damaged': class_names[np.argmax(score)], 'Score': 100*np.max(score)}
    except:
        return {"error": "Server Issues! Please Try Again Later!"}


@app.route('/severity', methods=["POST"])
def get_severity():
    try:
        # File Input
        file = request.files['image']
        # Checking Supported Extension
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            return jsonify({'error': 'Unsupported file format type'})
        filename = os.path.splitext(file.filename)[0]
        filepath = './Images/'+filename+'.jpg'
        file.save(filepath)
        # Accordingly Opening File
        with open(filepath, 'r') as file:
            print(type(file))
            if isinstance(file, str):
                image = load_img(filepath, target_size=(256, 256))
            elif isinstance(file, io.BytesIO) or isinstance(file, io.TextIOWrapper):
                image = Image.open(filepath)
                image = image.resize((256, 256))
            else:
                return jsonify({'error': 'Unsupported file object type'})
        # Loading Model
        model_name = request.form['model_name']
        model = load_model('./Models/bestModels/{}'.format(model_name))
        class_names = ["minor", "moderate", "severe"]
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
        os.remove(filepath)
        return {'Damaged': class_names[np.argmax(score)], 'Score': 100*np.max(score)}
    except:
        return {"error": "Server Issues! Please Try Again Later!"}


if __name__ == '__main__':
    app.run()
