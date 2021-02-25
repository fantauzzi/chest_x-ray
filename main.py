import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from pathlib import Path
import pickle
import random
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras_utils_orig import checkpointed_fit
import shutil

seed = 42
random.seed(seed)

comp_label = 'small'

comp_state_root = './computation_states'
comp_state_dir = comp_state_root + '/' + comp_label
models_root_dir = './models'
models_dir = models_root_dir + '/' + comp_label
dataset_root = '/mnt/storage/datasets/vinbigdata'
train_dir = dataset_root + '/train'
aug_dir = dataset_root + '/aug-train'  # Where generated augmented images will be stored
logs_root = './logs'  # Logs for Tensorboard will be organized in subdirs. within this dir.
metadata_pickle_fname = dataset_root + '/metadata.pickle'
base_trials_fname = comp_state_dir + '/base_model_trials.pickle'
ft_trials_fname = comp_state_dir + '/ft_model_trials.pickle'

''' For input resolution, check here, based on the choice of EfficientNet:
 https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ '''
input_res = (224, 224)
n_classes = 1  # Samples will be classified among the n_classes most frequent classes in the dataset
assert n_classes <= 14
# Can use the whole dataset, setting limit_samples to None, or a subset of it, of size limit_samples.
limit_samples = 200
# List of batch sizes for training to choose from during hyper-parameters tuning
train_batch_sizes = [8, 16, 24, 32, 48, 64]
val_batch_size = 64  # Batch size used for validation and also test
''' Number of experiments to be run for tuning of hyper-parameters of the base model and of fine tuning 
respectively'''
n_trials_base = 3  # Number of experiments to be run
n_trials_ft = 3
epochs_base_model = 5  # Max number of epochs with the base network frozen (for transfer learning)
epochs_ft = 5  # Max number of epochs for fine-tuning
epochs_cv = 5
''' Set to true to display all the (augmented or not) training samples in a grid, one line per batch. Samples
are shown the way they would be just before being fed to the model. Should be used in conjunction with 
limit_samples'''
n_folds = 5
vis_batches = False
print_base_model_summary = False
p_test = .15  # Proportion of the dataset to be held out for test (test set)
model_choice = tf.keras.applications.EfficientNetB0


def get_preprocessed_file_names(file_names):
    res = file_names.apply(
        lambda file_name: aug_dir + '/' + Path(file_name).stem + '_00.png')
    return res


def load_sample(file_name, *params):
    # TODO consider returning the image in the range [0,1] already, and change the way it is fed to the model
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image)

    # Perform all transformations on the image in the domain of floating point numbers [.0, 1.] , for improved accuracy
    image = tf.cast(image, tf.float32)

    # Maximise the image dynamic range, scaling the image values to the full interval [0-255]
    the_min = tf.reduce_min(image)
    the_max = tf.reduce_max(image)
    image = (image - the_min) / (the_max - the_min) * 255

    return (image, file_name) + params


def augment_image(image, file_name, aug_file_name, y, augment_params):
    side = float(tf.shape(image)[0])
    image_np = image.numpy()
    fill_mode = 'constant'
    cval = 0
    order = 1
    [zoom_x, zoom_y, shear, translate_x, translate_y] = augment_params
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   zx=zoom_x,
                                                                   zy=zoom_y,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   shear=shear,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image_np = tf.keras.preprocessing.image.apply_affine_transform(image_np,
                                                                   tx=translate_x * side,
                                                                   ty=translate_y * side,
                                                                   order=order,
                                                                   fill_mode=fill_mode,
                                                                   cval=cval)
    image = tf.convert_to_tensor(image_np)

    return image, file_name, aug_file_name, y, augment_params


def finalize_and_save_sample(image, file_name, aug_file_name, y=None, augment_params=None):
    # Make the image square, padding with 0s as necessary
    image_shape = tf.shape(image)
    target_side = tf.math.maximum(image_shape[0], image_shape[1])
    image = tf.image.resize_with_crop_or_pad(image, target_height=target_side, target_width=target_side)

    # Re-sample the image to the expected input size and type for the model
    image = tf.image.resize(image, size=input_res, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.math.round(image)
    image = tf.cast(image, tf.uint8)
    image_encoded = tf.io.encode_png(image, compression=-1)
    tf.io.write_file(aug_file_name, image_encoded)

    return aug_file_name


def make_aug_pipeline(file_names, y, augment_params, aug_file_names):
    dataset = tf.data.Dataset.from_tensor_slices((file_names, aug_file_names, y, augment_params))
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda *params: tf.py_function(func=augment_image,
                                       inp=params,
                                       Tout=[tf.float32, tf.string, tf.string, tf.int64, tf.float64]
                                       ),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(finalize_and_save_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def make_pre_process_pipeline(file_names, preprocess_file_names):
    dataset = tf.data.Dataset.from_tensor_slices((file_names, preprocess_file_names))
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.map(finalize_and_save_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def compute_sample_weights(metadata, variable_labels):
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

    int_class = metadata[variable_labels].apply(convert_to_int_class, axis=1)
    int_class_counts = int_class.value_counts()
    metadata_count = int_class.apply(lambda item: int_class_counts[item])
    metadata['weights'] = (sum(int_class_counts) - metadata_count) / sum(int_class_counts)

    print(f'\nFound {len(int_class_counts)} class combinations while assigning weights to training samples.')
    print(f'Most occurring class combination in training dataset appears {int_class_counts.max()} times.')
    print(f'Least occurring class combination in training dataset appears {int_class_counts.min()} time(s).')
    return metadata['weights']


def load_image(file_name, *params):
    # TODO consider returning the image in the range [0,1] already, and change the way it is fed to the model
    image = tf.io.read_file(file_name)
    image = tf.io.decode_png(image)
    image = tf.image.grayscale_to_rgb(image)
    return (image, file_name) + params


def make_pipeline(file_names, y, weights, batch_size, shuffle, for_model):
    if weights is None:
        weights = pd.Series([1.] * len(file_names))

    dataset = tf.data.Dataset.from_tensor_slices((file_names, y, weights))
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y), seed=42, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ''' If the dataset is meant to be fed to a model (for training or inference) then strip the information on the 
    file name '''
    if for_model:
        dataset = dataset.map(lambda *sample: (sample[0], sample[2], sample[3]),
                              num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def load_grayscale_image(file_name, *params):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_png(image)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.
    return (image,) + params

    # Instantiate the base model (pre-trained) for transfer learning


def make_pre_trained_model(pre_trained_model, dataset_mean, dataset_var, bias_init):
    # For additional regularization, try setting drop_connect_rate here below
    inputs = tf.keras.layers.Input(shape=input_res + (3,))
    pre_trained = pre_trained_model(input_tensor=inputs,
                                    input_shape=input_res + (3,),
                                    weights='imagenet',
                                    include_top=False,
                                    pooling='avg')

    pre_trained.layers[2].mean.assign(dataset_mean)
    pre_trained.layers[2].variance.assign(dataset_var)

    # Freeze the weights in the base model, to use it for transfer learning
    pre_trained.trainable = False

    # Append a final classification layer at the end of the base model (this will be trainable)
    x = pre_trained(inputs)
    x = Dense(1280 // 2, activation='sigmoid', name='dense_1')(x)
    outputs = Dense(n_classes, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias_init),
                    name='dense_final')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, pre_trained


def main():
    # str_time = datetime.datetime.now().strftime("%m%d-%H%M%S")

    aug_param_labels = ['zoom_x', 'zoom_y', 'shear', 'translate_x', 'translate_y']

    print(f'Input resolution {input_res}\nValidation batch size {val_batch_size}')

    Path(comp_state_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Load the datasets metadata in dataframes
    metadata = pd.read_csv(Path(dataset_root + '/train.csv'))
    # test.csv is for a Kaggle competition, no GT available. No need to load it here.
    # test_metadata = pd.read_csv(Path(dataset_root + '/test.csv'))

    """ Process the dataset to assign, to each x-ray, one or more diagnosis, based on a majority vote among the 
    radiologists. If at least 2 radiologists out of 3 have given the same diagnosis to a given x-ray, then the 
    diagnosis is assigned to the x-ray. If at least 2 radiologists out of 2 have assigned 'no finding' to the
    x-ray, then the x-ray is assigned 'no finding'. If there is no consensus on any diagnosis nor on 'no finding', then 
    set 'no finding' to 1. As a consequence, each x-ray is either assigned 'no finding' or at least one diagnosis. """

    grouped = metadata.groupby(['image_id', 'class_id'])['rad_id'].value_counts()
    images_ids = pd.Index(metadata['image_id'].unique())  # Using a pd.Index to speed-up look-up of values
    n_samples = len(images_ids)
    # Will hold the ground truth to be used for model training, the diagnosis for every x-ray image
    y = np.zeros((n_samples, 15), dtype=int)

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

    # Remove columns related to classes that won't be used (if any)
    metadata_df.drop(classes_by_freq[:len(classes_by_freq) - n_classes], axis=1, inplace=True)

    ''' Rebuild the column for class 14, i.e. 'no finding': iff the sample doesn't belong to any of the classes, then
    it belongs to class 14. Note that a 1 in this columns doesn't necessarily mean that radiologists gave a 
    'no finding' diagnose for the sample, it may be just lack of consensus.'''

    ''' Store the labels for the columns corresponding to dataset variables, will be used for training. These must
    not include column 14. Also rebuild column 14, as its content may have been invalidated by dropping columns
    related to classes that won't be used'''
    metadata_df.drop(14, axis=1, inplace=True)
    variable_labels = metadata_df.columns
    metadata_df[14] = pd.Series(metadata_df.sum(axis=1) == 0, dtype=int)

    # Add a column with the image file name
    metadata_df['image_id'] = train_dir + '/' + images_ids + '.jpg'

    # Limit the size of the dataset if required (variable `limit_samples`) and shuffle it.
    if limit_samples is None:
        metadata_df = metadata_df.sample(frac=1, random_state=seed)
    else:
        metadata_df = metadata_df.sample(n=limit_samples, random_state=seed)
    metadata_df.reset_index(inplace=True, drop=True)

    ''' Split the dataset into training, validation and test set; validation and test set contain the same number of 
    samples '''
    p_val = p_test / (1 - p_test)  # Proportion of the dataset not used for test to be used for validation
    metadata_train_val, metadata_test = train_test_split(metadata_df, test_size=p_test, random_state=seed,
                                                         stratify=metadata_df[14])
    metadata_train, metadata_val = train_test_split(metadata_train_val, test_size=p_val, random_state=seed,
                                                    stratify=metadata_train_val[14])

    ''' Now drop the column for class 14, 'no findig', as it is not needed anymore '''
    metadata_train = metadata_train.drop(14, axis=1)
    metadata_val = metadata_val.drop(14, axis=1)
    metadata_test = metadata_test.drop(14, axis=1)

    # sanity check
    assert len(metadata_train) + len(metadata_test) + len(metadata_val) == len(metadata_df)

    # Check for GPU presence
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU not found')
    else:
        print(f'Found GPU at: {device_name}')

    def make_augmented_metadata(sample):
        max_zoom = .15
        max_shear = 10
        max_translate_x = .05
        max_translate_y = .15
        augmented = sample
        augmented['zoom_x'] = 1 + random.uniform(-max_zoom, max_zoom)
        augmented['zoom_y'] = 1 + random.uniform(-max_zoom, max_zoom)
        augmented['shear'] = random.uniform(-max_shear, max_shear)
        augmented['translate_x'] = random.uniform(-max_translate_x, max_translate_x)
        augmented['translate_y'] = random.uniform(-max_translate_y, max_translate_y)
        ''' File name  of the augmented image is obtained from the original image file name, appending '_01' to its 
        stem; augmented images go to their own directory. '''
        augmented['aug_image_id'] = aug_dir + '/' + Path(sample['image_id']).stem + '_01.png'
        return augmented

    if Path(metadata_pickle_fname).is_file():
        print(f'\nLoading metadata for pre-processed and augmented detaset from file {metadata_pickle_fname}')
        with open(metadata_pickle_fname, 'rb') as pickle_f:
            pickled = pickle.load(pickle_f)
            metadata_train = pickled['metadata_train']
            metadata_val = pickled['metadata_val']
            metadata_test = pickled['metadata_test']
    else:
        # TODO consider vectorizing the operation below (e.g. with numpy), instead of repeating it on every df row
        print('\nMaking and saving pre-processed images for training, validation and test.')
        for metadata in (metadata_train, metadata_val, metadata_test):
            preprocess_file_names = get_preprocessed_file_names(metadata['image_id'])
            preprocess_pipeline = make_pre_process_pipeline(file_names=metadata['image_id'],
                                                            preprocess_file_names=preprocess_file_names)
            for _ in tqdm(preprocess_pipeline):
                pass
        print('Making and saving augmented images for training.')
        metadata_aug = metadata_train.apply(make_augmented_metadata, axis=1)
        aug_pipeline = make_aug_pipeline(file_names=metadata_aug['image_id'],
                                         y=metadata_aug[variable_labels],
                                         augment_params=metadata_aug[aug_param_labels],
                                         aug_file_names=metadata_aug['aug_image_id'])
        for _ in tqdm(aug_pipeline):
            pass
        for metadata in (metadata_train, metadata_val, metadata_test):
            metadata['image_id'] = get_preprocessed_file_names(metadata['image_id'])

        metadata_aug.drop(aug_param_labels + ['image_id'], axis=1, inplace=True)
        metadata_aug.rename(columns={'aug_image_id': 'image_id'}, inplace=True)
        metadata_train = metadata_train.append(metadata_aug, ignore_index=True)
        metadata_train = metadata_train.sample(frac=1, random_state=seed)
        pickle_this = {'metadata_train': metadata_train, 'metadata_val': metadata_val, 'metadata_test': metadata_test}
        with open(metadata_pickle_fname, 'wb') as pickle_f:
            pickle.dump(pickle_this, pickle_f, pickle.HIGHEST_PROTOCOL)
        print(f'\nSaved metadata for pre-processed and augmented dataset in {metadata_pickle_fname}')
        # These are not needed anymore, can free memory
        del aug_pipeline, preprocess_pipeline

    print(f'Augmented training set contains {len(metadata_train)} samples.')
    print(f'Validation set contains {len(metadata_val)} samples.')
    print(f'Test set contains {len(metadata_test)} samples.')

    metadata_train['weights'] = compute_sample_weights(metadata_train, variable_labels)

    """
    # Show a couple images from a pipeline, along with their GT, as a sanity check
    n_cols = 4
    n_rows = ceil(16 / n_cols)
    metadata_samples = metadata_train.sample(n=16, random_state=42)
    dataset_samples = make_pipeline(file_names=metadata_samples['image_id'],
                                    y=metadata_samples[variable_labels],
                                    weights=None,
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
    plt.draw()
    plt.pause(.01)
    """

    """print()
    model.summary()
    if print_base_model_summary:
        pre_trained.summary()

    def print_trainable_layers(model):
        print('\nTrainable layers:')
        count = 0
        for layer in model.layers:
            if layer.trainable:
                print(layer.name)
                count += 1
        print(f'{count} out of {len(model.layers)} layers are trainable')

    print_trainable_layers(model)"""

    ''' Calculate mean and variance of each image channel across the training set, and set them in the normalization
    layer of the pre-trained model '''

    # Another (slower) way to compute mean and variance across the images, commented out
    """images = np.zeros(shape=(len(metadata_train), 224, 224), dtype=float)
    for i, file_name in enumerate(metadata_train['image_id']):
        images[i] = plt.imread(file_name)"""

    stats_pipeline = tf.data.Dataset.from_tensor_slices((metadata_train['image_id'],))
    stats_pipeline = stats_pipeline.map(load_grayscale_image, num_parallel_calls=tf.data.AUTOTUNE)
    stats_pipeline = stats_pipeline.prefetch(buffer_size=tf.data.AUTOTUNE)
    stats_pipeline = stats_pipeline.map(tf.reduce_sum, num_parallel_calls=tf.data.AUTOTUNE)
    overall_sum = stats_pipeline.reduce(tf.constant(0, dtype=tf.float32), lambda a, b: a + b)
    mean_by_pixel = overall_sum / (len(metadata_train) * input_res[0] * input_res[1])

    stats_pipeline = tf.data.Dataset.from_tensor_slices((metadata_train['image_id'],
                                                         [mean_by_pixel] * len(metadata_train)))
    stats_pipeline = stats_pipeline.map(load_grayscale_image, num_parallel_calls=tf.data.AUTOTUNE)
    stats_pipeline = stats_pipeline.prefetch(buffer_size=tf.data.AUTOTUNE)
    stats_pipeline = stats_pipeline.map(
        lambda image, mean_by_pixel: tf.reduce_sum(tf.math.square(image - mean_by_pixel)),
        num_parallel_calls=tf.data.AUTOTUNE)
    overall_sum = stats_pipeline.reduce(tf.constant(0, dtype=tf.float32), lambda a, b: a + b)
    var_by_pixel = overall_sum / (len(metadata_train) * input_res[0] * input_res[1] - 1)

    print(f'Computed mean {mean_by_pixel.numpy()} and variance {var_by_pixel.numpy()} for the training dataset.')
    mean_by_pixel = tf.concat([mean_by_pixel] * 3, axis=0)
    var_by_pixel = tf.concat([var_by_pixel] * 3, axis=0)

    def display_by_batch(dataset):
        batch_iter = iter(dataset)
        batches_count = 0
        batch_size = -1
        for batch in batch_iter:
            batch_size = max([batch_size, batch[0].shape[0]])
            batches_count += 1
        fig, axs = plt.subplots(batches_count, batch_size, figsize=(2 * batch_size, 2 * batches_count), dpi=109.28)
        batch_iter = iter(dataset)
        for i_batch, batch in enumerate(batch_iter):
            this_batch_size = batch[0].shape[0]
            for i_sample in range(batch_size):
                if i_sample < this_batch_size:
                    image = batch[0][i_sample]
                    gt = batch[1][i_sample].numpy()
                    weight = batch[2][i_sample].numpy()
                    axs[i_batch, i_sample].imshow(image)
                    x_label = str(gt) + ' ' + str(np.around(weight, 5))
                    axs[i_batch, i_sample].set_xlabel(x_label)
                axs[i_batch, i_sample].set_xticks([])
                axs[i_batch, i_sample].set_yticks([])
        plt.draw()
        plt.pause(.01)

    # if vis_batches:
    #    display_by_batch(dataset_train)

    def compile_model(model, learning_rate):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(multi_label=True, name='auc')])
        return model

    def train_and_validate(params):
        '''
        For every trial t, the best model goes into ./models/<comp label>/<base_model/ft_model>-trial<000t>.h5
        Logs are in ./logs/<comp label>/<base_model/ft_model>-trial<000t>
        Files to resume computation are in ./computation_states/<comp label>/:
            <base_model/ft_model>_trials.pickle <- the state of hyperopt, to resume tuning
            <base_model/ft_model>.h5   <- model of the last completed epoch, to resume training
            <base_model/ft_model>.h5.prev <- model of the second last completed epoch
            <base_model/ft_model>.pickle <- variables needed to resume computation that are not in the .h5 model
            <base_model/ft_model>.pickle.prev <- previous file with the same
            <base_model/ft_model>_best.h5   <- model with the best validation score up to the last completed epoch
        '''
        pre_trained_model_fname, metadata_mean, metadata_var, train_batch_size, epochs, learning_rate, file_name_stem, \
        metadata_train, metadata_val = [params[key] for key in
                                        ('pre_trained_model_fname', 'metadata_mean', 'metadata_var', 'train_batch_size',
                                         'epochs', 'learning_rate', 'file_name_stem', 'metadata_train', 'metadata_val')]
        trials = params.get('trials', None)
        trial_idx = 0 if trials is None else len(trials.trials) - 1
        # file_name_stem will be either 'base_model' or 'ft_model'
        """
        # Deduce the index of the last trial run (if any) from the names of the log directories
        matches = sorted(Path(logs_root + '/' + comp_label).glob(f'{file_name_stem}-*'))
        if not matches:
            trial_idx = 0
        else:
            last_file_name = str(matches[-1])
            trial_idx = int(last_file_name[-3:len(last_file_name)]) + 1"""
        log_dir = logs_root + '/' + comp_label + '/{}-{:04d}'.format(file_name_stem, trial_idx)

        if trials is None:
            print(f'\nRunning experiment with parameters:')
        else:
            print(f'\nRunning experiment {trial_idx} with parameters:')
        for key, value in params.items():
            if key not in ('trials', 'metadata_train', 'metadata_val'):
                print(f'{key} = {value}')

        print(f'\nLogging training and validation to directory {log_dir}')

        ''' If the file name for a pre-trained base model was provided, then load it and fine tune it.
        Otherwise instantiate a base model and train it'''
        if pre_trained_model_fname is None:  # TODO consider moving this logic inside checkpointed_fit()
            # Calculate the bias to be used for proper initialization of the model last layer
            y_train_freq = metadata_train[variable_labels].sum(axis=0) / len(metadata_train)
            if min(y_train_freq) == 0:
                print(
                    'Some of the variables are never set to 1 in the training dataset. Increase the number of training samples.')
                print(y_train_freq)
                exit(-1)
            bias_init = np.log(y_train_freq / (1 - y_train_freq)).to_numpy()
            model, _ = make_pre_trained_model(pre_trained_model=model_choice,
                                              dataset_mean=metadata_mean,
                                              dataset_var=metadata_var,
                                              bias_init=bias_init)
        else:
            print(
                f'\nReloading pre-trained model from file {pre_trained_model_fname}')
            model = tf.keras.models.load_model(pre_trained_model_fname)
            ''' Ensure all layers are trainable, as model may have been saved with frozen weights after first training
            and before fine-tuning '''
            for layer in model.layers:
                layer.trainable = True

        model = compile_model(model, learning_rate=learning_rate)

        # Make the pipelines for training and validation
        dataset_train = make_pipeline(file_names=metadata_train['image_id'],
                                      y=metadata_train[variable_labels],
                                      weights=metadata_train['weights'],
                                      batch_size=train_batch_size,
                                      shuffle=True,
                                      for_model=True)

        dataset_val = make_pipeline(file_names=metadata_val['image_id'],
                                    y=metadata_val[variable_labels],
                                    weights=None,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    for_model=True) \
            if metadata_val is not None else None

        ''' Train and validate the model. Here shuffle is set to False as the pipeline already shuffles the data if 
        needed '''

        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
        path_and_fname_stem = '{}/{}-{:04d}'.format(comp_state_dir, file_name_stem, trial_idx)
        history = checkpointed_fit(model=model,
                                   path_and_fname_stem=path_and_fname_stem,
                                   metric_key='val_auc' if metadata_val is not None else None,
                                   optimization_direction='max',
                                   patience=100,
                                   max_epochs=epochs,
                                   alpha=learning_rate,
                                   decay=1.,
                                   k=1.,
                                   x=dataset_train,
                                   epochs=epochs,
                                   validation_data=dataset_val,
                                   shuffle=False,
                                   verbose=1,
                                   callbacks=[tensorboard_cb])

        if metadata_val is None:
            best_epoch_metric = None
        else:
            # Careful: keep argmax/argmin below set properly, based on the optimization direction of the metric
            best_epoch = np.argmax(history.history['val_auc'])
            # assert (history.epoch[best_epoch] == best_epoch)
            best_epoch_metric = history.history['val_auc'][best_epoch]

            ''' Take the model as optimized at the end of the epoch with the best validation score and make it
            available in the models_dir directory '''
        best_model_for_trial_fname = '{}/{}-trial{:04d}.h5'.format(models_dir, file_name_stem, trial_idx)
        shutil.copyfile(f'{path_and_fname_stem}_best.h5', best_model_for_trial_fname)

        # Hyperopt will minimize the loss below
        res = {'status': STATUS_OK,
               'loss': -best_epoch_metric if best_epoch_metric is not None else None,
               'history': history.history,
               'model_file_name': best_model_for_trial_fname}
        return res

    print('\nOptimizing and validating base (pre-trained) model.')
    if Path(base_trials_fname).is_file():
        with open(base_trials_fname, 'br') as pickle_f:
            print(f'Loading previously saved trials for base model from file {base_trials_fname}')
            trials = pickle.load(pickle_f)
            print(f'Already completed {len(trials.trials)} trials out of {n_trials_base}.')
    else:
        print(f'Trials for the base model will be saved in file {base_trials_fname}')
        trials = Trials()

    params_space = {'pre_trained_model_fname': None,
                    'metadata_mean': mean_by_pixel,
                    'metadata_var': var_by_pixel,
                    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-3)),
                    'train_batch_size': hp.choice('train_batch_size', train_batch_sizes),
                    'epochs': epochs_base_model,
                    'file_name_stem': 'base_model',
                    'metadata_train': metadata_train,
                    'metadata_val': metadata_val,
                    'trials': trials}

    best = fmin(fn=train_and_validate,
                space=params_space,
                algo=tpe.suggest,
                max_evals=n_trials_base,
                trials=trials,
                show_progressbar=False,
                trials_save_file=base_trials_fname,
                rstate=np.random.RandomState(seed))

    best_trial_idx = np.argmin([result['loss'] for result in trials.results])
    best_base_model_fname = trials.results[best_trial_idx]['model_file_name']

    print('\nFine-tuning and validating model.')
    if Path(ft_trials_fname).is_file():
        with open(ft_trials_fname, 'br') as pickle_f:
            print(f'Loading previously saved trials for fine-tuned model from file {ft_trials_fname}')
            trials_ft = pickle.load(pickle_f)
            print(f'Already completed {len(trials_ft.trials)} trials out of {n_trials_ft}.')
    else:
        print(f'Trials for fine-tuning of the model will be saved in file {ft_trials_fname}')
        trials_ft = Trials()

    params_space_ft = {'pre_trained_model_fname': best_base_model_fname,
                       'metadata_mean': mean_by_pixel,
                       'metadata_var': var_by_pixel,
                       'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-3)),
                       'train_batch_size': hp.choice('train_batch_size', train_batch_sizes),
                       'epochs': epochs_ft,
                       'file_name_stem': 'ft_model',
                       'metadata_train': metadata_train,
                       'metadata_val': metadata_val,
                       'trials': trials_ft}

    best_ft = fmin(fn=train_and_validate,
                   space=params_space_ft,
                   algo=tpe.suggest,
                   max_evals=n_trials_ft,
                   trials=trials_ft,
                   show_progressbar=False,
                   trials_save_file=ft_trials_fname,
                   rstate=np.random.RandomState(seed))

    best_trial_idx_ft = np.argmin([result['loss'] for result in trials_ft.results])
    ft_model_fname = trials_ft.best_trial['result']['model_file_name']
    print(
        f'\nBest fine-tuned model is in file {ft_model_fname}, result of experiment {best_trial_idx_ft + 1}/{n_trials_ft}')

    # Merge toghether training and validation set in one training set (dev set), and partition it for k-folds x-validation
    metadata_dev = metadata_train.append(metadata_val)
    metadata_dev['weights'] = compute_sample_weights(metadata_dev, variable_labels)
    metadata_dev = metadata_dev.sample(frac=1, random_state=seed)
    print(f'Dev. set for cross-validation of final model contains {len(metadata_dev)} samples.')

    # Make a Series that will be used for stratified partitioning of the dev set, making k-folds
    for_stratification = pd.Series(metadata_dev[variable_labels].sum(axis=1) == 0, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    training_folds, validation_folds = [], []
    for training, validation in skf.split(X=metadata_dev, y=for_stratification):
        training_folds.append(training)
        validation_folds.append(validation)

    print(f'\nStarting {n_folds}-fold cross-validation of the selected model over the dev. set.')
    # TODO set this properly
    params_dev = {'pre_trained_model_fname': ft_model_fname,
                  'metadata_mean': mean_by_pixel,
                  'metadata_var': var_by_pixel,
                  'learning_rate': best_ft['learning_rate'],
                  'train_batch_size': train_batch_sizes[best_ft['train_batch_size']],
                  'epochs': epochs_cv,
                  'trials': None}
    folds_history = None
    for k in range(n_folds):
        print(f'\nFold {k} of cross-validation')
        params_dev['file_name_stem'] = 'cross_validated_{:02d}'.format(k)
        params_dev['metadata_train'] = metadata_dev.iloc[training_folds[k]]
        params_dev['metadata_val'] = metadata_dev.iloc[validation_folds[k]]
        res = train_and_validate(params_dev)
        res['history']['fold'] = [k] * len(res['history']['loss'])
        res_df = pd.DataFrame(res['history'])
        folds_history = res_df if folds_history is None else folds_history.append(res_df)

    # folds_history['epoch'] = folds_history.index
    folds_history.reset_index(inplace=True)
    folds_history.rename(mapper={'index': 'epoch'}, inplace=True, axis=1)
    folds_averages = folds_history.groupby('epoch').mean().drop(labels='fold', axis=1)
    best_cv_epoch = np.argmax(folds_averages['val_auc'])

    print('Retraining it on the whole trainig+validation dataset, for testing and inference.')

    params_dev = {'pre_trained_model_fname': ft_model_fname,
                  'metadata_mean': mean_by_pixel,
                  'metadata_var': var_by_pixel,
                  'learning_rate': best_ft['learning_rate'],
                  'train_batch_size': train_batch_sizes[best_ft['train_batch_size']],
                  'epochs': best_cv_epoch + 1,
                  'file_name_stem': 'final_model',
                  'trials': None,
                  'metadata_train': metadata_dev,
                  'metadata_val': None}

    res = train_and_validate(params_dev)
    print(
        f"Final model re-trained with loss {res['history']['loss'][-1]} and AUC {res['history']['auc'][-1]}")

    dataset_eval = make_pipeline(metadata_test['image_id'],
                                 y=metadata_test[variable_labels],
                                 weights=None,
                                 batch_size=val_batch_size,
                                 shuffle=False,
                                 for_model=True)

    log_dir = logs_root + '/' + comp_label + '/final_model_test'
    print(f"\nTesting final model saved in {res['model_file_name']}. Logging to directory {log_dir}")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    histogram_freq=1,
                                                    profile_batch=0)

    final_model = tf.keras.models.load_model(res['model_file_name'])
    test_result = final_model.evaluate(dataset_eval,
                                     callbacks=[tensorboard_cb],
                                     return_dict=True,
                                     verbose=1)
    print(test_result)


if __name__ == '__main__':
    main()

''' TODO:
When starting the fine-tuning, it shouldn't resume from epoch 0, but from the latest one
Try a decaying learning rate (learning rate annealing)
Save report in csv format
Re-enable visualization of batches (say, first 4 samples of 4 batches)
Verify that setting all weights to 1 is the same as no weights, also loss-wise
Try caching at the end of the pipeline to fix the TB profiler issue, and let fit() do the shuffling
Do k-fold x-validation on the final model
Verify proper metrics for multi-label problems
Make the pipelines deterministic, check if any performance hit
Convert images from grayscale to RGB in batches
Try detection
Try another NN (e.g. DenseNet) and also train it from scratch (no transfer learning)  
'''
