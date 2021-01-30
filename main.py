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


def load_images(file_names, resolution, n_classes, dir, pickle_file=None):
    images = np.zeros(shape=(len(file_names), resolution[0], resolution[1]), dtype=np.uint8)
    if pickle_file is not None and Path(pickle_file).is_file():
        print(f'\nReading pickled images from file {pickle_file}')
        with open(pickle_file, 'rb') as the_file:
            from_pickle = pickle.load(the_file)
        if not from_pickle['file_names'].equals(file_names) or from_pickle['dir'] != dir or from_pickle[
            'resolution'] != resolution or n_classes != from_pickle['n_classes']:
            print(
                f'Data in pickle file {pickle_file} don\'t match data that are needed. To rebuild the pickle file, delete it and re-run the program')
            exit(-1)
        images = from_pickle['images']
        print(f'Read {len(images)} images from the file.')
        return images

    for i in tqdm.trange(len(file_names)):
        name = file_names.iloc[i]
        # TODO try with matplotlib and tf.image.resize()
        pil_image = Image.open(dir + '/' + name)
        pil_image = pil_image.resize(resolution, resample=3)
        image = np.array(pil_image, dtype=np.uint8)
        images[i] = image

    if pickle_file is not None:
        pickle_this = {'file_names': file_names, 'dir': dir, 'resolution': resolution, 'images': images,
                       'n_classes': n_classes}
        print(f'\nPickling images in file {pickle_file}')
        with open(pickle_file, 'wb') as pickle_file:
            pickle.dump(pickle_this, pickle_file, pickle.HIGHEST_PROTOCOL)
        print(f'Pickled {len(images)} images in the file.')
        return images


if __name__ == '__main__':
    ''' For input resolution, check here, based on the choice of EfficientNet:
     https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ '''
    input_res = (224, 224)
    n_classes = 3  # Samples will be classified among the n_variables most frequent classes in the dataset
    train_batch_size = 16
    val_batch_size = 16
    ''' The number of samples from the dataset to be used for training, validation and test of the model. Set to None
    to use all of the available samples. Used to make quick experiments on a small subset of the datasert '''
    limit_samples = None
    epochs = 25  # Number of epochs to run with the base network frozen (for transfer learning)
    epochs_ft = 100  # Number of epochs to run for fine-tuning
    model_choice = tf.keras.applications.EfficientNetB0

    print(
        f'Input resolution {input_res}\nTraining batch size {train_batch_size}\nValidation batch size {val_batch_size}')
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
    y = np.zeros((n_samples, 15),
                 dtype=int)  # The ground truth, i.e. the diagnosis for every x-ray image    y_df['weights'] = samples_weight

    for ((image_id, class_id, rad_id), _) in grouped.iteritems():
        y[images_ids.get_loc(image_id), class_id] += 1

    # Set to 1 entries for which there is consensus among at least two radiologists
    y = (y >= 2).astype(int)
    # If for any sample there is no consensus on any diagnosis, then set 'no finding' to 1 for that sample
    y[y.sum(axis=1) == 0, 14] = 1

    # Sort classes based on their frequency in the dataset, ascending order; but leave out the class meaning 'no finding'
    y_ones_count = y.sum(axis=0)
    classes_by_freq = np.argsort(y_ones_count[:14])

    # Sanity check
    count = 0
    for line in y:
        assert (sum(line[:14]) == 0 or line[-1] == 0)  # sum(line[:14]) > 0 => line[-1]==0
        assert (sum(line) > 0)  # Cannot be all zeroes: if there are no diagnosis, then 'no finding' must be set to 1
        assert ((line[-1] == 0 and sum(line[:14]) >= 0) or (line[-1] == 1 and sum(line[:14]) == 0))
        count += 1

    # Convert the obtained ground truth to a dataframe, and add a column with the image file names
    y_df = pd.DataFrame(y)

    # Remove columns related to classes that won't be used
    y_df.drop(classes_by_freq[:len(classes_by_freq) - n_classes], axis=1, inplace=True)

    ''' Rebuild the column for class 14, i.e. 'no finding': if the sample doesn't belong to any of the classes, then
    ensure it belongs to class 14 '''
    y_df.drop(14, axis=1, inplace=True)
    y_df[14] = pd.Series(y_df.sum(axis=1) == 0, dtype=int)

    # Compute sample weights, and add them as a new column to the dataframe
    totals = y_df.sum(axis=0)
    totals = totals.multiply(-1)
    totals = totals.add(len(y_df))
    samples_weights = y_df.dot(totals)
    samples_weights = pd.Series(samples_weights.divide((n_classes + 1) * len(y_df)), dtype=float)
    y_df['weights'] = samples_weights

    # Add a column with the image file name
    y_df['image_id'] = images_ids + '.jpg'

    # Limit the size of the dataset if required (variable `limit_samples`) and shuffle it.
    if limit_samples is None:
        y_df = y_df.sample(frac=1, random_state=42)
    else:
        y_df = y_df.sample(n=limit_samples, random_state=42)
    y_df.reset_index(inplace=True, drop=True)

    ''' Split the dataset into training, validation and test set; validation and test set contain the same number of 
    samples '''
    p_test = .15  # Proportion of the dataset to be used as test set
    p_val = p_test / (1 - p_test)  # Proportion of the dataset not used for test to be used for validation
    y_train_val, y_test = train_test_split(y_df, test_size=p_test, random_state=42, stratify=y_df[14])
    stratify_train_val = y_df[14].loc[y_train_val[14].index]
    y_train, y_val = train_test_split(y_train_val, test_size=p_val, random_state=42, stratify=y_train_val[14])

    # Now drop the column for class 14, 'no findig', as it is not needed anymore
    y_train = y_train.drop(14, axis=1)
    y_val = y_val.drop(14, axis=1)
    y_test = y_test.drop(14, axis=1)

    # sanity check
    assert len(y_train) + len(y_test) + len(y_val) == len(y_df)

    # Load all the images of the training dataset and store them in memory, for visualization and training
    images_train = load_images(y_train['image_id'],
                               resolution=input_res,
                               n_classes=n_classes,
                               dir=dataset_root + '/train',
                               pickle_file='logs/train_images.pickle')

    # File names of images are not needed in the dataframe anymore, drop them. Also, pop the series with weights from it
    weights = y_train['weights']
    y_train.drop(columns=['image_id', 'weights'], inplace=True)


    def preprocess_sample(image, y, weight=None):
        # TODO: move this at the bacth level for efficiency
        ''' Note: the Keras pre-trained EfficientNet normalizes the images itself, assumes the input images are encoded
        in the range [0, 255] '''
        # Turn grayscale images into 3 channel images, as that is what the model expects as input
        image = tf.stack([image, image, image], axis=2)
        return (image, y) if weight is None else (image, y, weight)


    def make_pipeline(x, y, weights, batch_size, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((x, y)) if weights is None else tf.data.Dataset.from_tensor_slices(
            (x, y, weights))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(y_train), seed=42, reshuffle_each_iteration=True)
        dataset = dataset.map(preprocess_sample)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


    # Show a couple images from a pipeline, along with their GT, as a sanity check
    n_cols = 4
    n_rows = ceil(16 / n_cols)
    dataset_samples = make_pipeline(x=images_train, y=y_train, weights=None, batch_size=n_cols * n_rows, shuffle=True)
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

    # Instantiate the base model (pre-trained) for transfer learning
    # For additional regularization, try setting drop_connect_rate here below
    inputs = tf.keras.layers.Input(shape=input_res + (3,))
    pre_trained = model_choice(input_tensor=inputs,
                               input_shape=input_res + (3,),
                               weights='imagenet',
                               include_top=False,
                               pooling='avg')

    # Freeze the weights in the base model, to use it for transfer learning
    pre_trained.trainable = False

    # Append a final classification layer at the end of the base model (this will be trainable)
    x = pre_trained(inputs)
    # TODO consider adding here batch normalization and then drop-out
    outputs = Dense(n_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    pre_trained.summary()


    def print_trainable_layers(model):
        print('\nTrainable layers:')
        count = 0
        for layer in model.layers:
            if layer.trainable:
                print(layer.name)
                count += 1
        print(f'{count} out of {len(model.layers)} layers are trainable')


    print_trainable_layers(model)
    print()


    def compile_model(model, learning_rate):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(multi_label=True, name='auc')])
        return model


    model = compile_model(model, learning_rate=3e-4)

    images_val = load_images(y_val['image_id'],
                             resolution=input_res,
                             n_classes=n_classes,
                             dir=dataset_root + '/train',
                             pickle_file='logs/val_images.pickle')
    y_val.drop(columns=['image_id', 'weights'], inplace=True)

    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'\nLogging in directory {log_dir}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    dataset_train = make_pipeline(x=images_train, y=y_train, weights=weights, batch_size=train_batch_size, shuffle=True)
    dataset_val = make_pipeline(x=images_val, y=y_val, weights=None, batch_size=val_batch_size, shuffle=True)


    def compute_normalization_params(dataset):
        batch_iter = iter(dataset)
        total_mean = tf.Variable((0, 0, 0), dtype=tf.float32)
        total_variance = tf.Variable((0, 0, 0), dtype=tf.float32)
        batches_count = 0
        for batch in batch_iter:
            mean = tf.math.reduce_mean(tf.cast(batch[0], dtype=tf.float32) / 255., axis=[0, 1, 2])
            variance = tf.math.reduce_variance(tf.cast(batch[0], dtype=tf.float32) / 255., axis=[0, 1, 2])
            total_mean.assign_add(mean)
            total_variance.assign_add(variance)
            batches_count += 1
        total_mean.assign(total_mean / batches_count)
        total_variance.assign(total_variance / batches_count)
        return total_mean, total_variance


    mean, variance = compute_normalization_params(dataset_train)
    print(f'\nComputed mean {mean} and variance {variance} for the training dataset.')
    pre_trained.layers[2].mean.assign(mean)
    pre_trained.layers[2].variance.assign(variance)

    history = model.fit(dataset_train,
                        epochs=epochs,
                        validation_data=dataset_val,
                        # class_weight=class_weight,
                        shuffle=False,  # Shuffling is already done on the dataset before it is being fed to fit()
                        callbacks=[tensorboard_callback],
                        verbose=1)

    # Un-freeze the weights of the base model to fine tune. Also see https://bit.ly/2YnJwqg
    print('\nFine-tuning the model.')
    pre_trained.trainable = True

    print_trainable_layers(model)
    print()

    # Because of the change in frozen layers, need to recompile the model
    model = compile_model(model, learning_rate=3e-5)

    log_dir_ft = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'\nLogging in directory {log_dir_ft}')
    tensorboard_callback_ft = tf.keras.callbacks.TensorBoard(log_dir=log_dir_ft, histogram_freq=1)

    history_ft = model.fit(dataset_train,
                           epochs=epochs_ft,
                           validation_data=dataset_val,
                           shuffle=False,  # Shuffling is already done on the dataset before it is being fed to fit()
                           callbacks=[tensorboard_callback_ft],
                           verbose=1)

    ''' TODO:
    Initialize last bias properly
    Maximize images dynamic range
    Image augmentation
    Deeper final classifier
    Take pre-processing at batch level
    Try to simplify doing single class classification, possibly also detection 
    '''
