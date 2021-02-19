#!/usr/bin/env python
# coding: utf-8

# # Training a neural network on MNIST with Keras
# 
# This simple example demonstrate how to plug TFDS into a Keras model.
# 

# Copyright 2020 The TensorFlow Datasets Authors, Licensed under the Apache License, Version 2.0

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/datasets/keras_example"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/keras_example.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/datasets/docs/keras_example.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import pickle
import shutil

# ## Step 1: Create your input pipeline
# 
# Build efficient input pipeline using advices from:
# * [TFDS performance guide](https://www.tensorflow.org/datasets/performances)
# * [tf.data performance guide](https://www.tensorflow.org/guide/data_performance#optimize_performance)
# 

# ### Load MNIST
# 
# Load with the following arguments:
# 
# * `shuffle_files`: The MNIST data is only stored in a single file, but for larger datasets with multiple files on disk, it's good practice to shuffle them when training.
# * `as_supervised`: Returns tuple `(img, label)` instead of dict `{'image': img, 'label': label}`

# In[2]:


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# ### Build training pipeline
# 
# Apply the following transormations:
# 
# * `ds.map`: TFDS provide the images as tf.uint8, while the model expect tf.float32, so normalize images
# * `ds.cache` As the dataset fit in memory, cache before shuffling for better performance.<br/>
# __Note:__ Random transformations should be applied after caching
# * `ds.shuffle`: For true randomness, set the shuffle buffer to the full dataset size.<br/>
# __Note:__ For bigger datasets which do not fit in memory, a standard value is 1000 if your system allows it.
# * `ds.batch`: Batch after shuffling to get unique batches at each epoch.
# * `ds.prefetch`: Good practice to end the pipeline by prefetching [for performances](https://www.tensorflow.org/guide/data_performance#prefetching).

# In[3]:


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# ### Build evaluation pipeline
# 
# Testing pipeline is similar to the training pipeline, with small differences:
# 
#  * No `ds.shuffle()` call
#  * Caching is done after batching (as batches can be the same between epoch)

# In[4]:


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# ## Step 2: Create and train the model
# 
# Plug the input pipeline into Keras.

# In[5]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


def keep_last_two_files(file_name):
    if Path(file_name).is_file():
        Path(file_name).replace(file_name + '.prev')
    Path(file_name + '.tmp').replace(file_name)


class CheckpointEpoch(tf.keras.callbacks.Callback):
    def __init__(self,
                 file_name_stem,
                 metric_key,
                 optimization_direction,
                 patience,
                 already_computed_epochs,
                 max_epochs,
                 best_so_far=None,
                 best_epoch=None):
        super(CheckpointEpoch, self).__init__()
        assert optimization_direction in ['min', 'max']
        assert already_computed_epochs < max_epochs
        self.optimization_direction = optimization_direction
        self.metric_key = metric_key
        self.patience = patience
        self.best_so_far = best_so_far if best_so_far is not None else float(
            'inf') if optimization_direction == 'min' else float('-inf')
        # self.epochs_total = already_computed_epochs  # Epochs are numbered starting from 1
        self.best_epoch = best_epoch
        self.model_file_name = None
        self.file_name_stem = file_name_stem
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        # self.epochs_total += 1
        # Save the model at the end of the epoch, to be able to resume training from there
        tf.keras.models.save_model(model=self.model,
                                   filepath=self.file_name_stem + '.h5.tmp',
                                   save_format='h5')
        keep_last_two_files(self.file_name_stem + '.h5')

        ''' If the last epoch had the best validation so far, then copy the saved model in a dedicated file,
        that can be loaded and used for testing and inference '''
        metric_value = logs[self.metric_key]
        if (self.optimization_direction == 'max' and metric_value > self.best_so_far) or \
                (self.optimization_direction == 'min' and metric_value < self.best_so_far):
            self.best_so_far = metric_value
            self.best_epoch = epoch
            new_model_file_name = self.file_name_stem + '_best.h5'
            print(
                f'Best epoch so far {self.best_epoch} with {self.optimization_direction} {self.metric_key} = {self.best_so_far} -Saving model in file {new_model_file_name}')
            shutil.copyfile(self.file_name_stem + '.h5', self.file_name_stem + '_best.h5')

        if epoch + 1 >= self.max_epochs:  # epochs are numbered from 0
            print(f'\nStopping training as it has reached the maximum number of epochs {self.max_epochs}')
            self.model.stop_training = True
        elif self.patience is not None and epoch - self.best_epoch > self.patience:
            if self.patience == 0:
                print('\nStopping training has there has been no improvement since the previous epoch.')
            else:
                print(f'\nStopping training as there have been no improvements for more than {self.patience} epoch(s).')
            self.model.stop_training = True

        # Save those variables, needed to resume training from the last epoch, that are not saved with the model
        pickle_this = {'epochs_total': epoch + 1,  # Epochs are numbered from 0
                       'best_so_far': self.best_so_far,
                       'best_epoch': self.best_epoch}
        pickle_fname = self.file_name_stem + '.pickle'
        with open(pickle_fname + '.tmp', 'bw') as pickle_f:
            pickle.dump(pickle_this, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        keep_last_two_files(pickle_fname)


class LearningRateScheduler():
    def __init__(self, alpha, decay, k):
        self.alpha = alpha
        self.decay = decay
        self.k = k
    def update(self, epoch, lr):
        print(f'Learning rate was {lr} at epoch {epoch}')
        updated_lr = self.alpha*(self.decay**(epoch/self.k))
        print(f'Now it is {updated_lr}')
        return updated_lr

def checkpointed_fit(model,
                     checkpoint_fname_stem,
                     metric_key,
                     optimization_direction,
                     patience,
                     max_epochs,
                     alpha,
                     decay,
                     k,
                     **args):
    # These variables will be overwritten if a saved model and pickle file exist
    already_computed_epochs = 0
    best_so_far = None
    best_epoch = None

    checkpoint_fname = checkpoint_fname_stem + '.h5'
    if Path(checkpoint_fname).is_file():
        model = tf.keras.models.load_model(checkpoint_fname)
        with open(checkpoint_fname_stem + '.pickle', 'br') as pickle_f:
            pickled = pickle.load(pickle_f)
        already_computed_epochs = pickled['epochs_total']
        best_so_far = pickled['best_so_far']
        best_epoch = pickled['best_epoch']

        if already_computed_epochs >= max_epochs:
            print(f'\nThe model has already been trained for the maximum number of epochs {max_epochs}.')
            return None
        if already_computed_epochs - best_epoch > patience:
            print(
                f'\nModel training already stopped after {already_computed_epochs} epoch(s) because it exceeded {patience} epoch(s) without improvement on metric {metric_key}.')
            return None

        print(
            f'\nResuming training after epoch {already_computed_epochs}. So far the best model validation had {metric_key}={best_so_far} at epoch {best_epoch}.')

    checkpoint_epoch_cb = CheckpointEpoch(file_name_stem=checkpoint_fname_stem,
                                          metric_key=metric_key,
                                          optimization_direction=optimization_direction,
                                          patience=patience,
                                          already_computed_epochs=already_computed_epochs,
                                          max_epochs=max_epochs,
                                          best_so_far=best_so_far,
                                          best_epoch=best_epoch)


    scheduler = LearningRateScheduler(alpha=alpha, decay=decay, k=k)
    learning_rate_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.update, verbose=1)
    callbacks = args.get('callbacks', [])
    callbacks += [checkpoint_epoch_cb, learning_rate_scheduler_cb]
    args['callbacks'] = callbacks
    history = model.fit(initial_epoch=already_computed_epochs, **args)
    return history


checkpointed_fit(model=model,
                 checkpoint_fname_stem='./epoch_checkpoint',
                 metric_key='val_sparse_categorical_accuracy',
                 optimization_direction='max',
                 patience=100,
                 max_epochs=24,
                 alpha = .001,
                 decay=.97,
                 k = 2.4,
                 x=ds_train,
                 epochs=24,
                 validation_data=ds_test)
