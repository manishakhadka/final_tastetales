import os
import glob

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.layers import Input
from keras import models

# Using Keras layers
from keras.layers import (
    #     BatchNormalization,
    #     Conv2D,
    Flatten,
    #     Dropout,
    #     MaxPool2D,
    #     GlobalMaxPool2D,
)

# Using custom layers by subclassing tf.keras.layers.Layer
from lib.layers import (
    BatchNormalization,
    Conv2D,
    # Flatten,
    Dropout,
    MaxPool2D,
    GlobalMaxPool2D,
    Dense,
)

DATASET = 'UTKFace'
DEFAULT_TRAIN_TEST_SPLIT = 0.7
IMG_SIZE, CHANNEL = 128, 1

ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())

NUM_AGE_CATEGORIES = 6  # 0-17, 18-25, 26-35, 36-45, 46-55, 56+


class UTKModel(models.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Convolutional and pooling layers
        self.conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool_1 = MaxPool2D()

        self.conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool_2 = MaxPool2D()

        self.conv_3 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool_3 = MaxPool2D()

        self.conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool_4 = MaxPool2D()

        # Flatten and Dense layers
        self.flatten = Flatten()

        self.hidden1 = Dense(256, activation='relu')
        self.hidden2 = Dense(256, activation='relu')

        self.dropout = Dropout(0.3)

        # Output layer for age regression
        self.age_output = Dense(1, name='age_output', activation='relu')

        # Additional output layer for age category classification
        self.age_category_output = Dense(
            NUM_AGE_CATEGORIES, activation='softmax', name='age_category_output')

    def call(self, inputs, training=None):
        x = self.conv_1(inputs)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.pool_3(x)

        x = self.conv_4(x)
        x = self.pool_4(x)

        x = self.flatten(x)

        x1 = self.hidden1(x)
        x2 = self.hidden2(x)

        x1 = self.dropout(x1, training=training)
        x2 = self.dropout(x2, training=training)

        age = self.age_output(x1)
        age_category = self.age_category_output(x2)

        return {
            'age_output': age,
            'age_category_output': age_category

        }


def categorize_age(age):
    if age <= 17:
        return 0  # Minor
    elif age <= 25:
        return 1  # Young Adult
    elif age <= 35:
        return 2  # Adult
    elif age <= 45:
        return 3  # Middle-Aged
    elif age <= 55:
        return 4  # Senior
    else:
        return 5  # Elderly


def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), int(gender), int(race)
    except Exception as e:
        print("Error parsing file:", filepath, e)
        return None, None, None


def process_dataset(dataset_dir):
    print("Processing dataset from", dataset_dir)
    files = glob.glob(os.path.join(dataset_dir, "*.jpg"))

    data = list(map(parse_filepath, files))

    dataframe = pd.DataFrame(data)
    dataframe['file'] = files
    dataframe['age_category'] = dataframe[0].apply(categorize_age)
    dataframe.columns = ['age', 'gender', 'race', 'file', 'age_category']
    dataframe = dataframe.dropna()

    dataframe['age'] = dataframe['age'].astype(int)

    return dataframe


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), IMG_SIZE, IMG_SIZE, CHANNEL)
    return features
