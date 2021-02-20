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
from keras_utils import checkpointed_fit

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




checkpointed_fit(model=model,
                 checkpoint_fname_stem='./epoch_checkpoint',
                 metric_key='val_sparse_categorical_accuracy',
                 optimization_direction='max',
                 patience=100,
                 max_epochs=20,
                 alpha=.001,
                 decay=.97,
                 k=2.4,
                 x=ds_train,
                 epochs=20,
                 validation_data=ds_test)
