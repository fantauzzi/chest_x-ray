import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from time import time
import cv2
from threading import Thread


def load_pics(*names):
    for name in names:
        pil_image = Image.open(name)
        pil_image.resize((1024, 1024))
        image = np.array(pil_image)
        # image = plt.imread(str(name))

def load_all_pics(train_pics_names):
    count = 0
    images = [None] * 15000
    n_threads = 16
    start = time()
    threads = [None] * n_threads
    pics_per_thread = len(train_pics_names) // n_threads
    for i in range(n_threads):
        threads[i] = Thread(group=None, target=load_pics, name=None,
                            args=train_pics_names[i * pics_per_thread:(i + 1) * pics_per_thread], kwargs={},
                            daemon=None)
        threads[i].start()
    for i in range(n_threads):
        threads[i].join()
    """
    for name in train_pics_names:
        # pil_image = Image.open(name)
        # image = np.array(pil_image)
        # image = cv2.imread(str(name))
        image = plt.imread(str(name))
        assert np.array_equal(image[0], image[1])
        assert np.array_equal(image[1], image[2])
        # images[count] = image
        # print(count)
        if count == 1000:
            break
        count += 1
    """
    print('Elapsed:', time() - start)

    # train_pics = image_dataset_from_directory(dataset_root+'/train')
    # train_pics_gen = ImageDataGenerator

...


def main():
    # Load the datasets metadata in dataframes
    dataset_root = '/mnt/storage/datasets/vinbigdata'
    train = pd.read_csv(Path(dataset_root + '/train.csv'))
    test = pd.read_csv(Path(dataset_root + '/test.csv'))
    train_pics_names = list(Path(dataset_root + '/train').glob('*.jpg'))

    # Find images that appear the most often in the training metadata




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#TODO provare a caricare le immagini fingendo sia un dataset anche se non lo e', vedere se e' piu' veloce