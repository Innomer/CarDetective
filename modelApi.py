import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import keras
import tensorflow as tf
import numpy as np

def predict_price_textual():
    lr_drop = load('./Models/linear_regression_drop.joblib')
    rfr_drop = load('./Models/random_forest_drop.joblib')
    lr_price = load('./Models/linear_regression_price.joblib')
    rfr_price = load('./Models/random_forest_price.joblib')


def confirm_car():
    # model = load_model('./Models/resnet_50/')
    model = load_model('./Models/vgg16_model_carnocar')
    # test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     './Datasets/car/test_brand_and_model_2',
    #     seed=42,
    #     image_size=(180, 180),
    #     batch_size=32,
    # )
    # class_names = test_ds.class_names
    class_names=["Not Car","Car"]
    image = load_img('./25.jpg', target_size=[224, 224])
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = keras.applications.resnet.preprocess_input(image)
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    return {'Car':class_names[np.argmax(score)],'Score':100*np.max(score)}