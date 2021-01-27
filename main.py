import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import datetime
import tqdm
import pickle


def load_images(file_names, dir, resolution, pickle_file=None):
    # pickle_file= 'logs/images.pickle'
    images = np.zeros(shape=(len(file_names), resolution[0], resolution[1]), dtype=np.uint8)
    if pickle_file is not None and Path(pickle_file).is_file():
        print(f'\nReading pickled images from file {pickle_file}')
        with open(pickle_file, 'rb') as the_file:
            from_pickle = pickle.load(the_file)
        if not from_pickle['file_names'].equals(file_names) or from_pickle['dir'] != dir or from_pickle[
            'resolution'] != resolution:
            print(
                f'Data in pickle file {pickle_file} don\'t match data that are needed. To rebuild the pickle file, delete it and re-run the program')
            exit(-1)
        images = from_pickle['images']
        print(f'Read {len(images)} images from the file.')
        return images

    for i in tqdm.trange(len(file_names)):
        name = file_names.iloc[i]
        pil_image = Image.open(dir + '/' + name)
        pil_image = pil_image.resize(resolution, resample=3)
        image = np.array(pil_image, dtype=np.uint8)
        images[i] = image

    if pickle_file is not None:
        pickle_this = {'file_names': file_names, 'dir': dir, 'resolution': resolution, 'images': images}
        print(f'\nPickling images in file {pickle_file}')
        with open(pickle_file, 'wb') as pickle_file:
            pickle.dump(pickle_this, pickle_file, pickle.HIGHEST_PROTOCOL)
        print(f'Pickled {len(images)} images in the file.')
        return images


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_res = (224, 224)
    train_batch_size = 16
    val_batch_size = 16
    # Load the datasets metadata in dataframes
    dataset_root = '/mnt/storage/datasets/vinbigdata'
    metadata = pd.read_csv(Path(dataset_root + '/train.csv'))
    test_metadata = pd.read_csv(Path(dataset_root + '/test.csv'))  # This is for the Kaggle competition, no GT available

    """ Process the dataset to assign, to each x-ray, one or more diagnosis, based on a majority vote among the 
    radiologists. If at least 2 radiologists out of 3 have given a same diagnosis to a given x-ray, then the 
    diagnosis is assigned to the given x-ray. If at least 2 radiologists out of 2 have assigned 'no finding' to the
    x-ray, then the x-ray is assigned 'no finding'. If there is no consensus on any diagnosis nor on 'no finding', then 
    set 'no finding' to 1. As a consequence, each x-ray is either assigned 'no finding' or at least one diagnosis. """
    grouped = metadata.groupby(['image_id', 'class_id'])['rad_id'].value_counts()
    images_ids = pd.Index(metadata['image_id'].unique())  # Using a pd.Index to speed-up look-up of values
    n_samples = len(images_ids)
    y = np.zeros((n_samples, 15), dtype=int)  # The ground truth, i.e. the diagnosis for every x-ray image
    for ((image_id, class_id, rad_id), _) in grouped.iteritems():
        y[images_ids.get_loc(image_id), class_id] += 1

    # Set to 1 entries for which there is consensus among at least two radiologists
    y = (y >= 2).astype(int)
    # If for any sample there is no consensus on any diagnosis, then set 'no finding' to 1 for that sample
    y[y.sum(axis=1) == 0, 14] = 1

    # Sanity check
    count = 0
    for line in y:
        assert (sum(line[:14]) == 0 or line[-1] == 0)  # sum(line[:14]) > 0 => line[-1]==0
        assert (sum(line) > 0)  # Cannot be all zeroes: if there are no diagnosis, then 'no finding' must be set to 1
        assert ((line[-1] == 0 and sum(line[:14]) >= 0) or (line[-1] == 1 and sum(line[:14]) == 0))
        count += 1

    # Convert the obtained ground truth to a dataframe, and add a column with the image file names
    y_df = pd.DataFrame(y)
    y_df['image_id'] = images_ids + '.jpg'
    # Shuffle the rows of the dataframe (samples)
    # dataset = dataset[:32]
    # y_df = y_df.sample(n=64, random_state=42)
    y_df = y_df.sample(frac=1, random_state=42)
    y_df.reset_index(inplace=True, drop=True)

    ''' Split the dataset into training, validation and test set; validation and test set contain the same number of 
    samples '''
    p_test = .15  # Proportion of the dataset to be used as test set
    p_val = p_test / (1 - p_test)  # Proportion of the dataset non used for test to be used for validation
    y_train_val, y_test = train_test_split(y_df, test_size=p_test, random_state=42, stratify=y_df[14])
    stratify_train_val = y_df[14].loc[y_train_val[14].index]
    y_train, y_val = train_test_split(y_train_val, test_size=p_val, random_state=42, stratify=stratify_train_val)
    # sanity check
    assert len(y_train) + len(y_test) + len(y_val) == len(y_df)
    assert (y_train.sum(numeric_only=True) + y_test.sum(numeric_only=True) + y_val.sum(
        numeric_only=True)).equals(y_df.sum(numeric_only=True))

    images_train = load_images(y_train['image_id'], dir=dataset_root + '/train', resolution=input_res,
                               pickle_file='logs/train_images.pickle')
    y_train.drop(columns=['image_id'], inplace=True)
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train, y_train))


    def preprocess_sample(image, y):
        image = tf.dtypes.cast(image, tf.float32) / 255.
        # image = tf.repeat(image, repeats=3, axis=0)
        image = tf.stack([image, image, image], axis=2)
        return image, y


    dataset_train = dataset_train.shuffle(buffer_size=len(y_train), seed=42, reshuffle_each_iteration=True)
    dataset_train = dataset_train.map(preprocess_sample)

    # Show a couple images from the generator, along with their GT, as a sanity check
    n_cols = 4
    n_rows = ceil(16 / n_cols)
    dataset_samples = dataset_train.batch(n_cols * n_rows, drop_remainder=False)
    samples_iter = iter(dataset_samples)
    samples = next(samples_iter)
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
                                                       input_shape=input_res + (3,),
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

    dataset_train = dataset_train.batch(batch_size=train_batch_size, drop_remainder=False)
    images_val = load_images(y_val['image_id'], dir=dataset_root + '/train', resolution=input_res,
                             pickle_file='logs/val_images.pickle')
    y_val.drop(columns=['image_id'], inplace=True)
    dataset_val = tf.data.Dataset.from_tensor_slices((images_val, y_val))
    dataset_val = dataset_val.map(preprocess_sample)
    dataset_val = dataset_val.batch(batch_size=val_batch_size, drop_remainder=False)

    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(dataset_train,
                        epochs=4,
                        # steps_per_epoch=steps_per_train_epoch,
                        validation_data=dataset_val,
                        # validation_steps=steps_per_val_epoch,
                        # class_weight=class_weight,
                        shuffle=False,  # Shuffling is already done on the dataset before it is being fed to fit()
                        callbacks=[tensorboard_callback],
                        verbose=1)

