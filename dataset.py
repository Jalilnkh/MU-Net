import math
import numpy as np
import matplotlib.pyplot as plt
from os import walk
from os.path import join
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split(dir_path, test_size, img_width=640, img_height=400, random_state=None):
    path_list = get_samples_path_list(dir_path)
    X, Y = read_images(path_list, img_width, img_height)
    if(test_size == 0):
        return X, Y
    if(random_state):
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, Y, test_size=test_size)


def illustrate(X, Y, ncols=5):
    # Illustrate the train images and masks
    N = len(X)

    nrows = math.ceil(float(N) / ncols) * 2

    plt.figure(figsize=(20, 16))
    for i in range(N):
        x_index = 2 * int(i / ncols) * ncols + (i % ncols) + 1
        plt.subplot(nrows, ncols, x_index)

        plt.imshow(np.squeeze(X[i]))
        plt.title('Image #{}'.format(i))
        plt.axis('off')

        y_index = x_index + ncols
        plt.subplot(nrows, ncols, y_index)

        # We display the associated mask we just generated above with the training image
        plt.imshow(np.squeeze(Y[i]), aspect="auto")
        print(x_index, y_index)
        plt.title('Mask #{}'.format(i))
        plt.axis('off')

    plt.show()


def read_images(path_list, img_width, img_height):
    path_list = list(path_list)
    N = len(path_list)

    X = np.zeros((N, img_height, img_width, 1), dtype=np.uint8)

    Y = np.zeros((N, img_height, img_width, 1), dtype=np.uint8)

    for i, v in tqdm(enumerate(path_list), total=N):
        sampel_path, label_path = v
        img = imread(sampel_path)
        img = resize(img, (img_height, img_width, 1),
                     mode='constant', preserve_range=True)

        X[i] = img

        mask = np.load(label_path)
        mask = np.reshape(mask, (img_height, img_width, 1))
        Y[i] = mask

    return X, Y


def get_samples_path_list(dir_path):
    for root, _, files in walk(dir_path):
        for f in files:
            if("label" in f and ".npy" in f):
                s = f.replace("label_", "").replace(".npy", "")
                s = f"{s}.png"
                s = join(root, s)
                f = join(root, f)
                yield s, f


if __name__ == "__main__":
    # you can your own dataset path
    path = "~/data/"
    X_train, X_test, Y_train, Y_test = split(path, 0.01)
    illustrate(X_test, Y_test, ncols=3)
