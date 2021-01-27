import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from time import time
from threading import Thread
from sklearn.model_selection import train_test_split
from math import ceil
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import datetime
import tqdm


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


def load_all_images(file_names, dir, resolution):
    images = np.zeros(shape=(len(file_names), resolution[0], resolution[1]), dtype=np.uint8)
    mapping = {}
    for i in tqdm.trange(len(file_names)):
        name = file_names.iloc[i]
        pil_image = Image.open(dir+'/'+name)
        pil_image = pil_image.resize(resolution, resample=3)
        image = np.array(pil_image, dtype=np.uint8)
        images[i] = image
        mapping[name] = i
    return images, mapping



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_res = (224, 224)
    # Load the datasets metadata in dataframes
    dataset_root = '/mnt/storage/datasets/vinbigdata'
    train = pd.read_csv(Path(dataset_root + '/train.csv'))
    test = pd.read_csv(Path(dataset_root + '/test.csv'))  # This is for the Kaggle competition, no GT available
    # train_pics_names = list(Path(dataset_root + '/train').glob('*.jpg'))

    """ Process the dataset to assign, to each x-ray, one or more diagnosis, based on a majority vote among the 
    radiologists. If at least 2 radiologists out of 3 have given a same diagnosis to a given x-ray, then the 
    diagnosis is assigned to the given x-ray. If at least 2 radiologists out of 2 have assigned 'no finding' to the
    x-ray, then the x-ray is assigned 'no finding'. If there is no consensus on any diagnosis nor on 'no finding', then 
    set 'no finding' to 1. As a consequence, each x-ray is either assigned 'no finding' or at least one diagnosis. """
    grouped = train.groupby(['image_id', 'class_id'])['rad_id'].value_counts()
    images_ids = pd.Index(train['image_id'].unique())  # Using a pd.Index to speed-up look-up of values
    n_samples = len(images_ids)
    diagnosis = np.zeros((n_samples, 15), dtype=int)
    for ((image_id, class_id, rad_id), _) in grouped.iteritems():
        diagnosis[images_ids.get_loc(image_id), class_id] += 1

    # Set to 1 entries for which there is consensus among at least two radiologists
    diagnosis = (diagnosis >= 2).astype(int)
    # If for any sample there is no consensus on any diagnosis, then set 'no finding' to 1 for that sample
    diagnosis[diagnosis.sum(axis=1) == 0, 14] = 1

    # Sanity check
    count = 0
    for line in diagnosis:
        assert (sum(line[:14]) == 0 or line[-1] == 0)  # sum(line[:14]) > 0 => line[-1]==0
        assert (sum(line) > 0)  # Cannot be all zeroes: if there are no diagnosis, then 'no finding' must be set to 1
        assert ((line[-1] == 0 and sum(line[:14]) >= 0) or (line[-1] == 1 and sum(line[:14]) == 0))
        count += 1

    # Convert the obtained metadata to a dataframe
    dataset = pd.DataFrame(diagnosis)
    dataset['image_id'] = images_ids + '.jpg'
    # Shuffle the rows of the dataframe (samples)
    # dataset = dataset.sample(frac=1, random_state=42)
    # dataset = dataset[:32]
    dataset = dataset.sample(n=32, random_state=42)
    images, image_name_to_row = load_all_images(dataset['image_id'], dir=dataset_root+'/train', resolution=input_res)

    ''' Split the dataset into training, validation and test set; validation and test set contain the same number of 
    samples '''
    p_test = .15  # Proportion of the dataset to be used as test set
    p_val = p_test / (1 - p_test)  # Proportion of the dataset non used for test to be used for validation
    dataset_train_val, dataset_test = train_test_split(dataset, test_size=p_test, random_state=42, stratify=dataset[14])
    stratify_train_val = dataset[14].loc[dataset_train_val[14].index]
    dataset_train, dataset_val = train_test_split(dataset_train_val, test_size=p_val, random_state=42,
                                                  stratify=stratify_train_val)
    # sanity check
    assert len(dataset_train) + len(dataset_test) + len(dataset_val) == len(dataset)
    assert (dataset_train.sum(numeric_only=True) + dataset_test.sum(numeric_only=True) + dataset_val.sum(
        numeric_only=True)).equals(dataset.sum(numeric_only=True))


    def preprocess_train(image):
        """min_val = min(image)
        max_val = max(image)
        return (image - min_val )/ (max_val - min_val)"""
        return image / 255


    def make_generator(dataframe, preprocessing_function, batch_size, shuffle):

        img_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)

        # train_batch_size = 32
        generator = img_gen.flow_from_dataframe(dataframe=dataframe,
                                                directory=dataset_root + '/train',
                                                target_size=input_res,
                                                # color_mode='grayscale',
                                                x_col='image_id',
                                                y_col=list(range(15)),
                                                class_mode='raw',
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                seed=42)
        return generator


    samples_generator = make_generator(dataframe=dataset,
                                       preprocessing_function=preprocess_train,
                                       batch_size=16,
                                       shuffle=True)
    # Show a couple images from the generator, along with their GT, as a sanity check
    n_cols = 4
    n_rows = ceil(16 / n_cols)
    samples = next(samples_generator)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(9.45, 4 * n_rows), dpi=109.28)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].imshow(samples[0][idx], cmap='gray')
            label = ''
            for pos in range(len(samples[1][idx])):
                if samples[1][idx][pos] == 1:
                    label = label + str(pos) + ' '
            idx += 1
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_xlabel(label)
    plt.draw()
    plt.pause(.01)

    pre_trained = tf.keras.applications.EfficientNetB5(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=None,
                                                       input_shape=input_res+(3,),
                                                       pooling='avg')

    predictions = Dense(15, activation='sigmoid')(pre_trained.output)

    model = Model(inputs=pre_trained.input, outputs=predictions)
    # for layer in model.layers[:len(model.layers)-1]:
    #    layer.trainable = False
    model.summary()
    print('\nTrainable layers:')
    for layer in model.layers:
        if layer.trainable:
            print(layer.name, layer)
    print()

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),  # Consider others, e.g. RMSprop
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')])

    train_generator = make_generator(dataframe=dataset_train,
                                     preprocessing_function=preprocess_train,
                                     batch_size=16,
                                     shuffle=True)

    val_generator = make_generator(dataframe=dataset_val,
                                   preprocessing_function=preprocess_train,
                                   batch_size=16,
                                   shuffle=True)

    steps_per_train_epoch = ceil(train_generator.samples / train_generator.batch_size)
    steps_per_val_epoch = ceil(val_generator.samples / val_generator.batch_size)
    last_batch_train_size = train_generator.samples - train_generator.batch_size * (steps_per_train_epoch - 1)
    last_batch_val_size = val_generator.samples - val_generator.batch_size * (steps_per_val_epoch - 1)
    print(
        f'\n{steps_per_train_epoch} batches, of {train_generator.batch_size} samples each, per training epoch; with an odd batch of size {last_batch_train_size}.')
    print(
        f'{steps_per_val_epoch} batches, of {val_generator.batch_size} samples each, per validation epoch; with an odd batch of size {last_batch_val_size}.')

    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_generator,
                        epochs=100,
                        steps_per_epoch=steps_per_train_epoch,
                        validation_data=val_generator,
                        validation_steps=steps_per_val_epoch,
                        # class_weight=class_weight,
                        callbacks=[tensorboard_callback],
                        verbose=1)

    # TODO: continue with loading all images in memory experimenting with a small dataset