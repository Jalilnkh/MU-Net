import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from skimage.io import imread
from sklearn.model_selection import train_test_split
from pathlib import Path
from os.path import split, join, exists
from config import *


def take_files(directory_path, pattern):
    return Path(directory_path).rglob(pattern)


def x_of(path):
    directory_path = split(path)[0]
    file_name = split(path)[1]
    file_name = file_name.replace("label_", "").replace(".npy", ".png")
    return join(directory_path, file_name)


def train_set():
    X = []
    Y = []
    for y in take_files("./participant", "*.npy"):
        y = str(y)
        x = x_of(y)
        if exists(x):
            X.append(x)
            Y.append(y)
    return X, Y


def split_train_set(X, Y, split):
    total_size = len(X)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(
        X, test_size=valid_size, random_state=RANDOM_STATE)
    train_y, valid_y = train_test_split(
        Y, test_size=valid_size, random_state=RANDOM_STATE)

    train_x, test_x = train_test_split(
        train_x, test_size=test_size, random_state=RANDOM_STATE)
    train_y, test_y = train_test_split(
        train_y, test_size=test_size, random_state=RANDOM_STATE)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def test_set():
    X = []
    X_train, _ = train_set()
    for x in take_files("./participant", "*.png"):
        if str(x) not in X_train:
            X.append(str(x))
    return X


def read_image(path):
    x = imread(path)
    x = x / 255.0
    return x


def read_mask(path):
    mask = np.load(path)
    return mask


def agument(X, Y):
    seq_orig = iaa.Sequential(
        [iaa.Rotate((-30, 30)), iaa.Rotate((-30, 30))], random_order=False)
    seq_flip = iaa.Sequential([iaa.Fliplr(0.5), iaa.Rotate(
        (-30, 30)), iaa.Rotate((-30, 30))], random_order=False)
    for x, y in zip(X, Y):
        img = read_image(x)
        mask = read_mask(y)
        images_aug = seq_orig(images=[img, mask])
        plt.show(img)
        plt.show(mask)


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([IMG_WIDTH, IMG_HEIGHT, 1])
    y.set_shape([IMG_WIDTH, IMG_HEIGHT, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


X, Y = train_set()
agument(X, Y)
