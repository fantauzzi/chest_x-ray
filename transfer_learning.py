# %%
"""
##### Copyright 2019 The TensorFlow Authors.
"""

# %%
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
# @title MIT License
#
# Copyright (c) 2017 François Chollet                                                                                                                    # IGNORE_COPYRIGHT: cleared by OSS licensing
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_utils import resumable_fit, Trainer, EWA_LearningRateScheduler, count_weights, k_fold_resumable_fit
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from pathlib import Path
from hyperopt import tpe, hp

"""
In this tutorial, you will use a dataset containing several thousand images of cats and dogs. Download and extract a 
zip file containing the images, then create a `tf.data.Dataset` for training and validation using the 
`tf.keras.preprocessing.image_dataset_from_directory` utility. You can learn more about loading images in this 
[tutorial](https://www.tensorflow.org/tutorials/load_data/images).
"""

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
seed = 42
test_set_fraction = 3

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

"""
Show the first nine images and labels from the training set:
"""
class_names = validation_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE


def make_train_dataset_cb(**params):
    assert params.get('x') is None
    batch_size = params['batch_size']
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=IMG_SIZE)

    dataset_size = len(list(Path(train_dir).glob('**/*.jpg')))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=dataset_size, seed=42, reshuffle_each_iteration=True)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    del (params['batch_size'])

    return train_dataset, params


train_dataset, ret_params = make_train_dataset_cb(batch_size=BATCH_SIZE)

assert not ret_params

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

"""
As the original dataset doesn't contains a test set, you will create one. To do so, determine how many batches of data are available in the validation set using ```tf.data.experimental.cardinality```, then move 20% of them to a test set.
"""

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // test_set_fraction)
validation_dataset = validation_dataset.skip(val_batches // test_set_fraction)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

"""
Configure the dataset for performance. Use buffered prefetching to load images from disk without having I/O become
blocking. To learn more about this method see the [data performance](https://www.tensorflow.org/guide/data_performance) 
guide.
"""

# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

"""
Use data augmentation. When you don't have a large image dataset, it's a good practice to artificially introduce 
sample diversity by applying random, yet realistic, transformations to the training images, such as rotation and 
horizontal flipping. This helps expose the model to different aspects of the training data and reduce 
[overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). You can learn more about data 
augmentation in this [tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).
"""

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

"""
Note: These layers are active only during training, when you call `model.fit`. They are inactive when the model is 
used in inference mode in `model.evaulate` or `model.fit`.
Let's repeatedly apply these layers to the same image and see the result.
"""

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

"""
Rescale pixel values

In a moment, you will download `tf.keras.applications.MobileNetV2` for use as your base model. This model expects 
pixel vaues in `[-1,1]`, but at this point, the pixel values in your images are in `[0-255]`. To rescale them, use 
the preprocessing method included with the model.
"""

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

"""
Note: Alternatively, you could rescale pixel values from `[0,255]` to `[-1, 1]` using a 
[Rescaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling) layer.
"""

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

"""
Note: If using other `tf.keras.applications`, be sure to check the API doc to determine if they expect pixels in 
`[-1,1]` or `[0,1]`, or use the included `preprocess_input` function.
Create the base model from the pre-trained convnets
You will create the base model from the **MobileNet V2** model developed at Google. This is pre-trained on the ImageNet
dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a
wide variety of categories like `jackfruit` and `syringe`. This base of knowledge will help us classify cats and dogs
from our specific dataset.

First, you need to pick which layer of MobileNet V2 you will use for feature extraction. The very last classification
layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful.  Instead, you
will follow the common practice to depend on the very last layer before the flatten operation. This layer is called the
"bottleneck layer". The bottleneck layer features retain more generality as compared to the final/top layer.

First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the 
**include_top=False** argument, you load a network that doesn't include the classification layers at the top, which is 
ideal for feature extraction.
"""

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

"""
This feature extractor converts each `160x160x3` image into a `5x5x1280` block of features. Let's see what it does to 
an example batch of images:
"""

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

"""
Feature extraction
In this step, you will freeze the convolutional base created from the previous step and to use as a feature extractor. 
Additionally, you add a classifier on top of it and train the top-level classifier.

Freeze the convolutional base

It is important to freeze the convolutional base before you compile and train the model. Freezing (by setting 
layer.trainable = False) prevents the weights in a given layer from being updated during training. MobileNet V2 has 
many layers, so setting the entire model's `trainable` flag to False will freeze all of them.
"""

base_model.trainable = False

"""
### Important note about BatchNormalization layers

Many models contain `tf.keras.layers.BatchNormalization` layers. This layer is a special case and precautions should 
be taken in the context of fine-tuning, as shown later in this tutorial. 

When you set `layer.trainable = False`, the `BatchNormalization` layer will run in inference mode, and will not update 
its mean and variance statistics. 

When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, you should keep the 
BatchNormalization layers in inference mode by passing `training = False` when calling the base model. Otherwise, 
the updates applied to the non-trainable weights will destroy what the model has learned.

For details, see the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).
"""

# Let's take a look at the base model architecture
# base_model.summary()

### Add a classification head

"""
To generate predictions from the block of features, average over the spatial `5x5` spatial locations, using a 
`tf.keras.layers.GlobalAveragePooling2D` layer to convert the features to  a single 1280-element vector per image.
"""

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

"""
Apply a `tf.keras.layers.Dense` layer to convert these features into a single prediction per image. You don't need an 
activation function here because this prediction will be treated as a `logit`, or a raw prediction value.  Positive 
numbers predict class 1, negative numbers predict class 0.
"""

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

"""
Build a model by chaining together the data augmentation, rescaling, base_model and feature extractor layers using the 
[Keras Functional API](https://www.tensorflow.org/guide/keras/functional). As previously mentioned, use training=False 
as our model contains a BatchNormalization layer.
"""

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

"""
### Compile the model
Compile the model before training it. Since there are two classes, use a binary cross-entropy loss with 
`from_logits=True` since the model provides a linear output.
"""

base_learning_rate = 0.0001


def compile_cb(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


compile_cb(model)
model.summary()

"""
The 2.5M parameters in MobileNet are frozen, but there are 1.2K _trainable_ parameters in the Dense layer.  These are 
divided between two `tf.Variable` objects, the weights and biases.
"""

"""
### Train the model
After training for 10 epochs, you should see ~94% accuracy on the validation set.
"""

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs/transfer_learning',
                                                histogram_freq=1,
                                                profile_batch=0)

learning_rate_cb = EWA_LearningRateScheduler(verbose=1)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath='./comp_state/transfer_learning_best.h5',
                                                   monitor='val_accuracy',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   save_freq='epoch')

train_batch_sizes = [16, 32, 64]

hp_space = {  # 'x': train_dataset,
    'epochs': initial_epochs,
    'batch_size': hp.choice('train_batch_size', train_batch_sizes),
    'validation_data': validation_dataset,
    # 'alpha': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-3)),
    'alpha': base_learning_rate,
    'decay': .97,
    'k': hp.uniform('k', 2., 6.),
    'callbacks': [tensorboard_cb, checkpoint_cb, learning_rate_cb]}

trainer = Trainer(model=model,
                  compile_cb=compile_cb,
                  comp_dir='./comp_state',
                  stem='transfer_learning',
                  space=hp_space,
                  val_metric='val_accuracy',
                  optimization_direction='max',
                  log_dir='./logs',
                  make_train_dataset_cb=make_train_dataset_cb)
# Note: if `space` only contains constants, no random variables to sample from, then `res` here below will be {}
res = trainer.do_it(max_evals=10, algo=tpe.suggest, show_progressbar=False, rstate=np.random.RandomState(seed))

best_trial = trainer.trials.best_trial
best_trial_history = best_trial['result']['history']
best_pretrained = best_trial['result']['model_file_name']
best_metric_value = max(best_trial_history[trainer.val_metric])
best_metric_epoch = np.argmax(best_trial_history[trainer.val_metric])
print(
    f'Best trained model saved in {best_pretrained} with {trainer.val_metric}={best_metric_value} obtained at epoch {best_metric_epoch}')

print('Best hyper-parameter values:')
for k, v in res.items():
    print(k, v)

"""
### Learning curves
Let's take a look at the learning curves of the training and validation accuracy/loss when using the MobileNet V2 base
model as a fixed feature extractor.
"""

best_trial_history = trainer.trials.best_trial['result']['history']
acc = best_trial_history['accuracy']
val_acc = best_trial_history['val_accuracy']

loss = best_trial_history['loss']
val_loss = best_trial_history['val_loss']
if False:
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

"""
Note: If you are wondering why the validation metrics are clearly better than the training metrics, the main factor 
is because layers like `tf.keras.layers.BatchNormalization` and `tf.keras.layers.Dropout` affect accuracy during 
training. They are turned off when calculating validation loss.

To a lesser extent, it is also because training metrics report the average for an epoch, while validation metrics are 
evaluated after the epoch, so validation metrics see a model that has trained slightly longer.
"""

"""
## Fine tuning
In the feature extraction experiment, you were only training a few layers on top of an MobileNet V2 base model. 
The weights of the pre-trained network were **not** updated during training.

One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the 
pre-trained model alongside the training of the classifier you added. The training process will force the weights 
to be tuned from generic feature maps to features associated specifically with the dataset.

Note: This should only be attempted after you have trained the top-level classifier with the pre-trained model set 
to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train 
all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the 
classifier) and your pre-trained model will forget what it has learned.

Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. In most 
convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple 
and generic features that generalize to almost all types of images. As you go higher up, the features are increasingly 
more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized 
features to work with the new dataset, rather than overwrite the generic learning.
"""

"""
### Un-freeze the top layers of the model
All you need to do is unfreeze the `base_model` and set the bottom layers to be un-trainable. Then, you should 
recompile the model (necessary for these changes to take effect), and resume training.
"""
model = tf.keras.models.load_model(best_pretrained, compile=False)
base_model = model.layers[4]
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

"""
### Compile the model
As you are training a much larger model and want to readapt the pretrained weights, it is important to use a lower 
learning rate at this stage. Otherwise, your model could overfit very quickly.
"""


def compile_fine_cb(model):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


compile_fine_cb(model)
model.summary()

"""
### Continue training the model
If you trained to convergence earlier, this step will improve your accuracy by a few percentage points.
"""

fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

checkpoint_cb_fine = tf.keras.callbacks.ModelCheckpoint(filepath='./comp_state/fine_best.h5',
                                                        monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        save_freq='epoch')

tensorboard_fine_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs/fine',
                                                     histogram_freq=1,
                                                     profile_batch=0)

hp_space_fine = {'epochs': fine_tune_epochs,
                 'batch_size': hp.choice('train_batch_size', train_batch_sizes),
                 'validation_data': validation_dataset,
                 # 'alpha': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-3)),
                 'alpha': base_learning_rate / 10,
                 'decay': .97,
                 'k': hp.uniform('k', 2., 6.),
                 'callbacks': [tensorboard_fine_cb, checkpoint_cb_fine, learning_rate_cb]}

trainer_fine = Trainer(model=model,
                       compile_cb=compile_fine_cb,
                       comp_dir='./comp_state',
                       stem='fine',
                       space=hp_space_fine,
                       val_metric='val_accuracy',
                       optimization_direction='max',
                       log_dir='./logs',
                       make_train_dataset_cb=make_train_dataset_cb)
# Note: if `space` only contains constants, no random variables to sample, then `res` here below will be {}
res = trainer_fine.do_it(max_evals=20, algo=tpe.suggest, show_progressbar=False, rstate=np.random.RandomState(seed))
print(res)

"""
Let's take a look at the learning curves of the training and validation accuracy/loss when fine-tuning the last few 
layers of the MobileNet V2 base model and training the classifier on top of it. The validation loss is much higher 
than the training loss, so you may get some overfitting.

You may also get some overfitting as the new training set is relatively small and similar to the original MobileNet V2 
datasets.
After fine tuning the model nearly reaches 98% accuracy on the validation set.
"""
best_trial = trainer_fine.trials.best_trial
best_trial_history = best_trial['result']['history']
best_pretrained = best_trial['result']['model_file_name']
best_metric_value = max(best_trial_history[trainer_fine.val_metric])
best_metric_epoch = np.argmax(best_trial_history[trainer_fine.val_metric])
print(
    f'Best trained model saved in {best_pretrained} with {trainer_fine.val_metric}={best_metric_value} obtained at epoch {best_metric_epoch}')

print('Best hyper-parameter values:')
for k, v in res.items():
    print(k, v)

best_trial_history = trainer_fine.trials.best_trial['result']['history']
acc += best_trial_history['accuracy']
val_acc += best_trial_history['val_accuracy']

loss += best_trial_history['loss']
val_loss += best_trial_history['val_loss']

model = tf.keras.models.load_model(best_pretrained)

best_k = res['k']
best_train_batch_size = train_batch_sizes[res['train_batch_size']]

# No need to compile the model, as k_fold_resumable_fit() will take care of it


def make_datasets_cb(fold, n_folds, **kwargs):
    dataset1 = image_dataset_from_directory(train_dir,
                                            shuffle=False,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
    dataset1_size = len(list(Path(train_dir).glob('**/*.jpg')))

    dataset2 = image_dataset_from_directory(validation_dir,
                                            shuffle=False,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
    dataset2_size = len(list(Path(validation_dir).glob('**/*.jpg')))
    dataset2 = dataset2.skip(int(
        np.ceil(
            dataset2_size / BATCH_SIZE)) // test_set_fraction)  # Skip the part of the validation dataset that is used for testing

    dataset_size = dataset1_size + dataset2_size - dataset2_size // test_set_fraction
    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.unbatch()
    dataset = dataset.cache()
    dataset = dataset.shuffle(dataset_size, seed=42, reshuffle_each_iteration=False)

    fold_size = (dataset_size) // n_folds
    batch_size = kwargs['batch_size']

    val_ds = dataset.skip(fold_size * fold)
    val_ds = val_ds.take(fold_size)
    val_ds = val_ds.batch(batch_size=batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds1 = dataset.take(fold_size * fold)
    train_ds2 = dataset.skip(fold_size * (fold + 1))
    train_ds2 = train_ds2.take(-1)
    train_ds = train_ds1.concatenate(train_ds2)
    train_ds = train_ds.shuffle(buffer_size=dataset_size, seed=43, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs/kfold',
                                                histogram_freq=1,
                                                profile_batch=0)

learning_rate_cb = EWA_LearningRateScheduler(alpha=base_learning_rate / 10, decay=.97, k=best_k)

def compile_cb(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


histories, means = k_fold_resumable_fit(model=model,
                                        comp_dir='./kfold_compstate',
                                        stem='transfer_learning',
                                        compile_cb=compile_cb,
                                        make_datasets_cb=make_datasets_cb,
                                        n_folds=4,
                                        epochs=fine_tune_epochs,
                                        batch_size=best_train_batch_size,
                                        callbacks=[tensorboard_cb, learning_rate_cb],
                                        log_dir='./logs/kfold')

# No need to compile the model, as resumable_fit() will take care of it
model = tf.keras.models.load_model(best_pretrained)


def make_dev_dataset(**kwargs):
    assert kwargs.get('x') is None
    batch_size = kwargs['batch_size']
    dataset1 = image_dataset_from_directory(train_dir,
                                            shuffle=False,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
    dataset1_size = len(list(Path(train_dir).glob('**/*.jpg')))

    dataset2 = image_dataset_from_directory(validation_dir,
                                            shuffle=False,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
    dataset2_size = len(list(Path(validation_dir).glob('**/*.jpg')))
    dataset2 = dataset2.skip(int(
        np.ceil(
            dataset2_size / BATCH_SIZE)) // test_set_fraction)  # Skip the part of the validation dataset that is used for testing

    dataset_size = dataset1_size + dataset2_size - dataset2_size // test_set_fraction
    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.unbatch()
    dataset = dataset.cache()
    dataset = dataset.shuffle(dataset_size, seed=42, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


dev_dataset = make_dev_dataset(batch_size=best_train_batch_size)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs/final',
                                                histogram_freq=1,
                                                profile_batch=0)

learning_rate_cb = EWA_LearningRateScheduler(alpha=base_learning_rate / 10, decay=.97, k=best_k)

history_dev = resumable_fit(model=model,
                            comp_dir='./final_comp_state',
                            stem='final',
                            compile_cb=compile_fine_cb,
                            x=dev_dataset,
                            epochs=fine_tune_epochs,
                            initial_epoch=initial_epochs,
                            validation_data=None,
                            callbacks=[tensorboard_cb, learning_rate_cb])

print(history_dev.history)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""
### Evaluation and prediction
Finaly you can verify the performance of the model on new data using test set.
"""

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

"""
And now you are all set to use this model to predict if your pet is a cat or dog.
"""

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
plt.imshow(image_batch[i].astype("uint8"))
plt.title(class_names[predictions[i]])
plt.axis("off")

"""
## Summary

* **Using a pre-trained model for feature extraction**:  When working with a small dataset, it is a common practice 
to take advantage of features learned by a model trained on a larger dataset in the same domain. This is done by 
instantiating the pre-trained model and adding a fully-connected classifier on top. The pre-trained model is "frozen" 
and only the weights of the classifier get updated during training.
In this case, the convolutional base extracted all the features associated with each image and you just trained a 
classifier that determines the image class given that set of extracted features.

* **Fine-tuning a pre-trained model**: To further improve performance, one might want to repurpose the top-level layers
of the pre-trained models to the new dataset via fine-tuning.
In this case, you tuned your weights such that your model learned high-level features specific to the dataset. 
This technique is usually recommended when the training dataset is large and very similar to the original dataset 
that the pre-trained model was trained on.

To learn more, visit the [Transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).
"""

""" TODO
Extract the test set from the x-validation dev set, make sure it is not used in training any model
"""
