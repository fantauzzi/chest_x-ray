import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from pathlib import Path
from math import ceil
import random

input_res = (224, 224)
n_classes = 3
limit_samples = None
p_test = .15

dataset_root = '/mnt/storage/datasets/vinbigdata'
metadata = pd.read_csv(Path(dataset_root + '/train.csv'))
# test.csv is for a Kaggle competition, no GT available
# test_metadata = pd.read_csv(Path(dataset_root + '/test.csv'))

""" Process the dataset to assign, to each x-ray, one or more diagnosis, based on a majority vote among the 
radiologists. If at least 2 radiologists out of 3 have given a same diagnosis to a given x-ray, then the 
diagnosis is assigned to the given x-ray. If at least 2 radiologists out of 2 have assigned 'no finding' to the
x-ray, then the x-ray is assigned 'no finding'. If there is no consensus on any diagnosis nor on 'no finding', then 
set 'no finding' to 1. As a consequence, each x-ray is either assigned 'no finding' or at least one diagnosis. """
grouped = metadata.groupby(['image_id', 'class_id'])['rad_id'].value_counts()
images_ids = pd.Index(metadata['image_id'].unique())  # Using a pd.Index to speed-up look-up of values
n_samples = len(images_ids)
y = np.zeros((n_samples, 15),
             dtype=int)  # The ground truth, i.e. the diagnosis for every x-ray image

for ((image_id, class_id, rad_id), _) in grouped.iteritems():
    y[images_ids.get_loc(image_id), class_id] += 1

# Set to 1 entries for which there is consensus among at least two radiologists
y = (y >= 2).astype(int)
# If for any sample there is no consensus on any diagnosis, then set 'no finding' to 1 for that sample
y[y.sum(axis=1) == 0, 14] = 1

""" Sort classes based on their frequency in the dataset, ascending order; but leave out the class meaning 
'no finding' """
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
metadata_df = pd.DataFrame(y)

# Remove columns related to classes that won't be used
metadata_df.drop(classes_by_freq[:len(classes_by_freq) - n_classes], axis=1, inplace=True)

''' Rebuild the column for class 14, i.e. 'no finding': iff the sample doesn't belong to any of the classes, then
it belongs to class 14. Note that a 1 in this columns doesn't necessarily mean that radiologists gave a 
'no finding' diagnose for the sample.'''

metadata_df.drop(14, axis=1, inplace=True)
''' Store the labels for the columns corresponding to dataset variables, will be used for training. These must
not include column 14. '''
variable_labels = metadata_df.columns
metadata_df[14] = pd.Series(metadata_df.sum(axis=1) == 0, dtype=int)

# Add a column with the image file name
metadata_df['image_id'] = dataset_root + '/samples/' + images_ids + '.jpg'

metadata_df.reset_index(inplace=True, drop=True)
metadata_df.loc[:8, 'image_id'] = pd.Series(
    [dataset_root + '/samples/' + name + '.jpg' for name in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')])

available_images = list(Path(dataset_root + '/samples/').glob('*.jpg'))

available_images = [str(item) for item in available_images]
metadata_train = metadata_df[metadata_df['image_id'].isin(available_images)]
metadata_train = metadata_train.sort_values(by=['image_id'], key=lambda item: item.str.lower())
''' Now drop the column for class 14, 'no findig', as it is not needed anymore '''
metadata_train = metadata_train.drop(14, axis=1)

""" Compute training sample weights, and add them as a new column to the dataframe. The sequence of labels (1 and 0) 
for each sample are interpreted as a binary number, which becomes a "class" for the given sample, for the 
purpose of calculating the sample weight. If the obtained "class" appears in the training dataset p times, and the 
dataset contains N samples overall, then the weight for the given samples is set to (N-p)/N.
TODO: could samples with very rare "class" get a weight so high to screw the gradient of the loss? """


def convert_to_int_class(row):
    int_value = 0
    for i, item in enumerate(row):
        pos = len(row) - i - 1
        int_value += item * (2 ** pos)
    return int_value


int_class = metadata_train[variable_labels].apply(convert_to_int_class, axis=1)
int_class_counts = int_class.value_counts()
metadata_count = int_class.apply(lambda item: int_class_counts[item])
metadata_train['weights'] = (sum(int_class_counts) - metadata_count) / sum(int_class_counts)

print(f'\nFound {len(int_class_counts)} class combinations while assigning weights to training samples.')
print(f'Most occurring class combination in training dataset appears {int_class_counts.max()} times.')
print(f'Least occurring class combination in training dataset appears {int_class_counts.min()} time(s).')


def load_image(x, y, weight):
    # TODO consider returning the image in the range [0,1] already, and change the way it is fed to the model
    image = tf.io.read_file(x)
    image = tf.io.decode_jpeg(image)

    # Perform all transformations on the image in the domain of floating point numbers [.0, 1.] , for improved accuracy
    image = tf.cast(image, tf.float32)

    # Maximise the image dynamic range, scaling the image values to the full interval [0-255]
    the_min = tf.reduce_min(image)
    the_max = tf.reduce_max(image)
    image = (image - the_min) / (the_max - the_min) * 255

    return image, x, y, weight


def augmentation(image, x, y, weight):
    image_np = image.numpy()
    fill_mode = 'constant'
    cval = 0
    order = 1
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   zx=1 / .75,
                                                                   zy=1 / .75,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   shear=-10,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   tx=-100,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image = tf.convert_to_tensor(image_np)

    # TODO consider moving the rest of this to a different function that can enter a computation graph

    # Make the image square, padding with 0s as necessary
    image_shape = tf.shape(image)
    target_side = tf.math.maximum(image_shape[0], image_shape[1])
    image = tf.image.resize_with_crop_or_pad(image, target_height=target_side, target_width=target_side)

    # Re-sample the image to the input size for the model
    image = tf.image.resize(image, size=input_res, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.math.round(image)
    image = tf.cast(image, tf.uint8)
    image = tf.image.grayscale_to_rgb(image)

    return image, x, y, weight




def make_pipeline(x, y, weights, batch_size, shuffle, for_model):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda *params: tf.py_function(func=augmentation,
                                       inp=params,
                                       Tout=[tf.uint8, tf.string, tf.int64, tf.float64]
                                       ),
        num_parallel_calls=tf.data.AUTOTUNE)  # TODO Tout parameter must change when there are weights
    # If the dataset is meant to be fed to a model (for training or inference) then strip the information on the file name
    if for_model:
        dataset = dataset.map(lambda *sample: (sample[0], sample[2], sample[3]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# Show a couple images from a pipeline, along with their GT, as a sanity check
n_cols = 4
n_rows = ceil(8 / n_cols)
weights = [1.] * len(metadata_train)
metadata_train['weights'] = weights


def augment_dataset(metadata):
    # Extend the dataset to have one augmented sample for every non-augmented sample
    def augment_sample(sample, augment):
        max_zoom = .15
        max_shear = 10
        max_translate_x = .05
        max_translate_y = .15
        augmented = sample
        if augment:
            augmented['augmented'] = True
            augmented['zoom_x'] = 1 + random.uniform(-max_zoom, max_zoom)
            augmented['zoom_y'] = 1 + random.uniform(-max_zoom, max_zoom)
            augmented['shear'] = random.uniform(-max_shear, max_shear)
            augmented['translate_x'] = random.uniform(-max_translate_x, max_translate_x)
            augmented['translate_y'] = random.uniform(-max_translate_y, max_translate_y)
        else:
            augmented['augmented'] = False
            augmented['zoom_x'] = 1.
            augmented['zoom_y'] = 1.
            augmented['shear'] = .0
            augmented['translate_x'] = .0
            augmented['translate_y'] = .0
        return augmented

    metadata_aug = metadata.apply(augment_sample, args=(False,), axis=1)
    metadata_aug2 = metadata.apply(augment_sample, args=(True,), axis=1)
    metadata_aug = metadata_aug.append(metadata_aug2, ignore_index=True)
    metadata_aug = metadata_aug.sample(frac=1, random_state=42)
    return metadata_aug


metadata_train = augment_dataset(metadata_train)

dataset_samples = make_pipeline(x=metadata_train['image_id'],
                                y=metadata_train[variable_labels],
                                weights=metadata_train['weights'],
                                batch_size=n_cols * n_rows,
                                shuffle=False,
                                for_model=False)
samples_iter = iter(dataset_samples)
samples = next(samples_iter)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=109.28)
idx = 0
for row in range(n_rows):
    for col in range(n_cols):
        axs[row, col].imshow(samples[0][idx], cmap='gray')
        x_label = Path(str(samples[1][idx])).stem
        y_label = ''
        for pos in range(samples[2].shape[1]):
            if samples[2][idx][pos] == 1:
                y_label = y_label + str(variable_labels[pos]) + ' '
        idx += 1
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        axs[row, col].set_xlabel(x_label)
        axs[row, col].set_ylabel(y_label)
# plt.draw()
# plt.pause(.01)
plt.show()
