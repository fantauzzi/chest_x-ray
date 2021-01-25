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




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the datasets metadata in dataframes
    dataset_root = '/mnt/storage/datasets/vinbigdata'
    train = pd.read_csv(Path(dataset_root + '/train.csv'))
    test = pd.read_csv(Path(dataset_root + '/test.csv'))  # This is for the Kaggle competition, no GT available
    # train_pics_names = list(Path(dataset_root + '/train').glob('*.jpg'))

    """ Process the dataset to assign, to each x-ray, one or more diagnosis, based on a majority vote among the 
    radiologists. If at least 2 radiologists out of 3 have given a same diagnosis to a given x-ray, then the 
    diagnosis is assigned to the given x-ray. If at least 2 radiologists out of 2 have assigned 'no finding' to the
    x-ray, then the x-ray is assigned 'no finding'. As a consequence, each x-ray is either assigned 'no finding', or
    one or more diagnosis. """
    grouped = train.groupby(['image_id', 'class_id'])['rad_id'].value_counts()
    images_ids = train.image_id.value_counts().index
    n_samples = len(images_ids)
    diagnosis = np.zeros((n_samples, 15), dtype=int)
    for ((image_id, class_id, rad_id), _) in grouped.iteritems():
        diagnosis[images_ids.get_loc(image_id), class_id] += 1
    diagnosis = (diagnosis >= 2).astype(int)

    # Sanity check
    count = 0
    for line in diagnosis:
        # sum(line[:14]) > 0 => line[-1]==0
        assert (sum(line[:14]) == 0 or line[-1] == 0)
        assert ((line[-1] == 0 and sum(line[:14]) >= 0) or (line[-1] == 1 and sum(line[:14]) == 0))
        count += 1

    # Convert the obtained metadata to a dataframe
    dataset = pd.DataFrame(diagnosis)
    dataset['image_id'] = images_ids
    # Shuffle the rows of the dataframe (samples)
    dataset = dataset.sample(frac=1)
    

