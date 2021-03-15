import pickle
import logging
from pathlib import Path
from functools import reduce
from copy import deepcopy
from time import time
from datetime import timedelta
from hyperopt import fmin, STATUS_OK, Trials
import tensorflow as tf
import numpy as np
import pandas as pd


def keep_last_two_files(file_name):
    if Path(file_name).is_file():
        Path(file_name).replace(file_name + '.prev')
    Path(file_name + '.tmp').replace(file_name)


class EWA_LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, alpha=3e-4, decay=1., k=1., verbose=0):
        super(EWA_LearningRateScheduler, self).__init__(schedule=self.update, verbose=verbose)
        self.alpha = alpha
        self.decay = decay
        self.k = k

    def update(self, epoch, lr):
        updated_lr = self.alpha * (self.decay ** (epoch / self.k))
        return updated_lr


def collect_trainable_states(layers):
    """
    Returns a list with the trainable state for all the given layers that have a trainable state. In case the
    layers have nested layers, then their states appear in the list as explored in pre-order traversal.
    :param layers: the given layers, a sequence. E.g. it could be attribute `layers` of an instance of `tf.keras.Model`.
    :return: a list of 2-tuples; the first element in every tuple is the name of the layer, the second element its
    trainable state. Note that layer names may not be unique.
    """
    res = []
    for layer in layers:
        if hasattr(layer, 'trainable'):
            res.append((layer.name, layer.trainable))
        if hasattr(layer, 'layers'):
            layer_res = collect_trainable_states(layer.layers)
            res.extend(layer_res)
    return res


def set_trainable_states(layers, names_and_state, idx=0):
    """
    Configure layers to have a given trainable state, as previously returned by `collect_trainable_state()`.
    :param layers: a sequence of layers the state must be applied to. E.g. it could be attribute `layers` of an
    instance of `tf.keras.Model`; in that case, after calling the function, the model should be compiled before
    optimizing it.
    :param names_and_state: a sequence of tuples as returned by `collect_trainable_state()`.
    :param idx: the parameter is used internally for recursion, traversing layers of layers; leave it to the default
    value (or set it to 0) when calling the function.
    :return: the number of layers to which a trainable state was applied (total of trainable and untrainable layers).
    """
    for layer in layers:
        if hasattr(layer, 'trainable'):
            name, layer.trainable = names_and_state[idx]
            assert layer.name == name
            idx = idx + 1
        if hasattr(layer, 'layers'):
            idx = set_trainable_states(layer.layers, names_and_state, idx)
    return idx


def count_components(tensors):
    """
    Returns the total number of components in a sequence of Tensorflow tensors. E.g. if a tensor has shape (3,2,1),
    its components are 3x2x1=6.
    :param tensors: the given sequence of tensors.
    :return: the number of components summed across all the tensors in the given sequence.
    """
    res = sum([np.prod(variable.shape) for variable in tensors])
    return res


def count_weights(layer_or_model):
    """
    Returns the count of trainable, non-trainable and total weights in a given model or layer. The count also includes
    all nested layers, if any.
    :param layer_or_model: a Keras layer or model.
    :return: a 3-tuple, with respectively the count of trainable, non-trainable and total weights.
    """
    trainables_weights, non_trainable_weights, total_weights = 0, 0, 0
    if hasattr(layer_or_model, 'trainable_weights'):
        trainables_weights += count_components(layer_or_model.trainable_weights)
    if hasattr(layer_or_model, 'non_trainable_weights'):
        non_trainable_weights += count_components(layer_or_model.non_trainable_weights)
    if hasattr(layer_or_model, 'weights'):
        total_weights += count_components(layer_or_model.weights)
    return trainables_weights, non_trainable_weights, total_weights


def make_pickle_file_name(filepath):
    # Find the last `.h5` or `.tf suffix`, and replace it with `.pickle`
    suffixes = Path(filepath).suffixes  # Suffixes includes the dot, e.g. 'model.h5' => '.h5'
    new_suffixes = []
    found = False
    for suffix in suffixes[::-1]:
        if suffix in ('.h5', '.tf') and not found:
            found = True
            new_suffixes.append('.pickle')
        else:
            new_suffixes.append(suffix)
    assert found
    prev_suffixes = ''.join(suffixes)
    new_suffixes = ''.join(new_suffixes[::-1])
    pickle_filepath = filepath[:len(filepath) - len(prev_suffixes)] + new_suffixes
    return pickle_filepath


def save_keras_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None,
                     signatures=None, options=None, save_traces=True):
    """
    Save a Keras model and the trainability state of each of its layers in two separate files. The former is
    saved through a call to
    [tf.keras.models.save_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model), see its
    parameters documentation. The trainability state is saved in a Python pickled file, with name obtained
    replacing the extension of the given `filepath` with '.pickle' (e.g. './my_model.h5' becomes './my_model.pickle').
    The model can then be reloaded, along with the saved trainability states, calling `load_keras_model()`.
    """
    pickle_fname = make_pickle_file_name(filepath)
    trainable = collect_trainable_states(model.layers)
    tf.keras.models.save_model(model, filepath, overwrite, include_optimizer, save_format, signatures, options,
                               save_traces)
    with open(pickle_fname, 'wb') as pickle_f:
        pickle.dump(trainable, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)


def load_keras_model(filepath, custom_objects=None, compile=True, options=None):
    """
    Load a Keras model previously saved via `save_keras_model()`. It loads the model by calling
    [tf.keras.models.load_model()](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model), see its
    parameters documentation. It also loads, and applies to the model, the trainability state for each model layer.
    If any layer is non-trainable, then the model needs to be compiled after it is returned by the function, even if
    parameter `compile` was set to True. That's because the trainability state are applied to the model after the call
    to `tf.keras.models.load_model()`.
    :return: a 2-tuple; the first item in the tuple is the Keras model, the second item is a boolean, True if at least
    one layer in the model is set to non-trainable, False otherwise. If the second item is True, then the model should
    be compiled before being optimized, even if parameter `compile` was set to True.
    """
    pickle_fname = make_pickle_file_name(filepath)
    model = tf.keras.models.load_model(filepath, custom_objects, compile, options)
    with open(pickle_fname, 'rb') as pickle_f:
        trainable = pickle.load(pickle_f)
    total_trainable = reduce(lambda partial_sum, item: partial_sum + item[1], trainable, 0)
    if total_trainable < len(trainable):
        set_trainable_states(model.layers, trainable)
        trainable_state_changed = True
    else:
        trainable_state_changed = False

    return model, trainable_state_changed


class CheckpointEpoch(tf.keras.callbacks.Callback):
    """
    Callback that saves the state of the computation at the
    """

    def __init__(self, comp_dir, stem, history):
        super(CheckpointEpoch, self).__init__()
        self.comp_dir = comp_dir
        self.stem = stem
        self.vars_fname = f'{self.comp_dir}/{self.stem}_vars.pickle'
        self.tmp_vars_fname = self.vars_fname + '.tmp'
        self.prev_vars_fname = self.vars_fname + '.prev'
        self.model_fname = f'{self.comp_dir}/{self.stem}_model.h5'
        self.tmp_model_fname = self.model_fname + '.tmp'
        self.prev_model_fname = self.model_fname + '.prev'
        self.history = {} if history is None else deepcopy(history)
        self.epochs_in_history = 0 if not self.history else len(next(iter(self.history.values())))
        self.model_checkpoint_cb = None

    def save_checkpoint(self):
        """
        Save the state of the computation in files, such that the computation can be resumed from that state.
        """
        ''' Some of the variables needed to restore the computation are not saved in a .h5 file; pickle them in a 
        separate file '''
        updated_history = {}
        for k, v in self.model.history.history.items():
            updated_history[k] = self.history.get(k, []) + self.model.history.history[k]
        pickle_this = {'history': updated_history,
                       'epoch': self.model.history.epoch,
                       'params': self.model.history.params}
        if self.model_checkpoint_cb is not None:
            pickle_this['epochs_since_last_save'] = self.model_checkpoint_cb.epochs_since_last_save
            pickle_this['best'] = self.model_checkpoint_cb.best
        with open(self.tmp_vars_fname, 'bw') as pickle_f:
            pickle.dump(pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        keep_last_two_files(self.vars_fname)
        # Save the model to be able to resume the computation. Note: save_format='tf' would be much slower
        save_keras_model(self.model, filepath=self.tmp_model_fname, save_format='h5')
        keep_last_two_files(self.model_fname)
        pickle_filepath = make_pickle_file_name(self.model_fname)
        keep_last_two_files(pickle_filepath)

    def on_train_begin(self, logs=None):
        Path(self.comp_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        """
        At the beginning of every epoch, except epoch 0, save the state of the computation as it was at the end of the
        previous epoch. This cannot be done conveniently in on_epoch_end() because, at that time, some information
        about the computation history in self.model.history.epoch has not been updated yet. State after the end of the
        last epoch of the computation will be saved by on_train_end()
        """
        if self.model.history.epoch:
            # Save a checkpoint with the previous epoch (the last epoch that completed)
            self.save_checkpoint()

    def on_train_end(self, logs=None):
        """
        Save the state of the computation at the end of its last epoch, and remove temporary files that are not needed
        anymore.
        """
        # Save a checkpoint with the last epoch of the training
        self.save_checkpoint()
        # Remove the .prev files of the computation, not needed anymore
        Path(self.prev_model_fname).unlink(missing_ok=True)
        pickle_filepath = make_pickle_file_name(self.prev_model_fname)
        Path(pickle_filepath).unlink(missing_ok=True)
        Path(self.prev_vars_fname).unlink(missing_ok=True)


def make_k_fold_file_name(stem):
    """
    Returns the pickle file name for a given stem.
    :param stem: the given stem, a string.
    :return: the pickle file name.
    """
    fname = stem + '_kfold.pickle'
    return fname


def k_fold_resumable_fit(model, comp_dir, stem, compile_cb, make_datasets_cb, n_folds, log_dir, log_level=logging.INFO,
                         **kwargs):
    """
    Performs k-fold cross validation of a Keras model, training and validating the model k times. The dataset
    partitioning in k folds is the responsibility of the client code, to be performed in a callback passed to the
    function, see parameter `make_datasets_cb`.

    :param model: the Keras model. In doesn't need to be compiled, as the function will take care of it. If a previous
    training and validation process on the same model has been interrupted, but its state was saved in files, then the
    function will load the model from files and resume training and validation, and this parameter will be ignored,
     it can be set to None.
    :param comp_dir: path to the directory where the training state will be saved, a string.
    :param stem: a stem that will be used to make file names to save the computation state and its results, a string.
    :param compile_cb: a function that will be called to build the Keras model as necessary. It must take one
    argument, which is the model to be compiled; anything it returns is ignored.
    :param make_datasets_cb: a function or method that instantiates the training and validation pipelines.
    It takes three parameters: the fold number, a non-negative integer, the total number of folds, a positive
    integer, and **kwargs, as passed to this function. It must return two instances of td.data.Dataset, with the
    training and validation datasets respectively for the given fold number.
    :param n_folds: number of folds (k) required for the k-fold cross-validation.
    :param log_dir: base name for the directory where to save logs for Tensorboard. Logs for fold XX are saved in
    <log_dir>-fold<nn>, where <nn> is the fold number. Logs for the overall validation, that is the average of the k
    validations, are saved in <log_dir>-xval.
    :param log_level: logging level for this funciton, as defined in package `logging`.
    :param kwargs: parameters to be passed to resumable_fit() for the training and validation of each fold.
    :return: a 2-tuple; the first element of the tuple is a list of k History objects, each with a record of the
    computation on fold k, as returned by tf.Keras.Model.fit(). The second element is a pd.DataFrame with the averages
    of validation metrics per epoch, averaged across folds.
    """
    assert kwargs.get('x') is None
    assert kwargs.get('validation_data') is None

    logger = make_logger(name='k_fold_resumable_fit', log_level=log_level)

    state_file_path = make_k_fold_file_name(f'{comp_dir}/{stem}')
    current_fold = 0
    histories = []

    # Restore the state of the k-fold cross validation from file, if the file is available
    if Path(state_file_path).is_file():
        with open(state_file_path, 'br') as pickle_f:
            pickled = pickle.load(pickle_f)
        current_fold = pickled['fold'] + 1
        histories = pickled['histories']
        logger.info(
            f"Reloaded the state of previous k-fold cross validation from {state_file_path} - {pickled['fold']} folds already computed")
    else:
        logger.info(f"State of k-fold cross validation will be saved in {state_file_path}")

    saved_model_fname = f'{comp_dir}/{stem}_orig.h5'

    for fold in range(current_fold, n_folds):
        if fold == 0:  # TODO move in common subroutine, to be shared with resumable_fit_wrapper()
            logger.info(
                f'Starting/resuming cross-validation of fold 0, saving model {model.name} at beginning of computation in {saved_model_fname}')
            compile_cb(model)
            save_keras_model(model, filepath=saved_model_fname, save_format='h5')
        else:
            logger.info(f'Starting/resuming cross-validation of fold {fold}, reloading model from {saved_model_fname}')
            model, _ = load_keras_model(saved_model_fname, compile=False)
            compile_cb(model)

        train_ds, val_ds = make_datasets_cb(fold, n_folds, **kwargs)
        kwargs['x'] = train_ds
        kwargs['validation_data'] = val_ds
        fold_stem = '{}-fold{:02d}'.format(stem, fold)
        logger.info(f'Processing fold {fold} - Total folds to be processed: {n_folds}')
        callbacks = kwargs.get('callbacks', [])
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.TensorBoard) and log_dir is not None:
                cb.log_dir = log_dir + '-fold{:02d}'.format(fold)
        history = resumable_fit(model=model, comp_dir=comp_dir, stem=fold_stem, compile_cb=compile_cb, **kwargs)
        histories.append(history.history)
        # Update the state of the k-fold x-validation as saved in the pickle
        with open(state_file_path + '.tmp', 'bw') as pickle_f:
            pickle_this = {'fold': fold, 'histories': histories}
            pickle.dump(obj=pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
            keep_last_two_files(state_file_path)

    logger.info(f"All {n_folds} folds of the cross-validation have been processed")
    histories_df = None
    for i, history in enumerate(histories):
        if histories_df is None:
            histories_df = pd.DataFrame(history)
            histories_df['fold'] = i
            histories_df['epoch'] = histories_df.index
        else:
            history_df = pd.DataFrame(history)
            history_df['fold'] = i
            history_df['epoch'] = history_df.index
            histories_df = pd.concat([histories_df, history_df], ignore_index=True)

    means = histories_df.groupby(['epoch']).mean()
    to_be_dropped = ['fold', 'lr']
    for column in means.columns:
        if str(column)[:4] != 'val_':
            to_be_dropped.append(column)
    means.drop(labels=to_be_dropped, axis=1, inplace=True)
    means['epoch'] = means.index

    file_writer = tf.summary.create_file_writer(log_dir + '-xval')
    # TODO do I need to set the default below?
    file_writer.set_as_default()
    for column in means.columns:
        for epoch, data in zip(means['epoch'], means[column]):
            tf.summary.scalar(str(column),
                              data=data,
                              step=epoch,
                              description='Average of validation metrics across folds during k-fold cross validation.')

    return histories, means


def resumable_fit(model, comp_dir, stem, compile_cb, **kwargs):
    """
    Trains, and optionally evaluate, a model, saving model and training state in files at the end of every epoch; can
    resume a previously interrupted training from the end of the last epoch that was completed.
    :param model: the model to be trained, a Keras model. If a previous training process on the same model has been
    interrupted, but its state was saved in files, then the function will load the model from files and resume training,
    and this parameter will be ignored and can be set to None. The function will compile the model, does not require the
    model to have been compiled already.
    :param comp_dir: path to the directory where the training state will be saved, a string.
    :param stem: a stem that will be used to make file names to save the computation state and its results, a string.
    :param compile_cb: a function that will be called to build the Keras model as necessary. It must take one
    argument, which is the model to be compiled; anything it returns is ignored.
    :param kwargs: keyword arguments to be passed to tf.Keras.Model.fit(). If there is a 'callbacks' argument listing
    an instance of tf.keras.callbacks.ModelCheckpoint, then the state of the latter is also saved and restored at
    the end of every epoch.
    :return: a History object with a record of the computation, as returned by tf.Keras.Model.fit(). In case of
    multiple calls to this function for the same model, to resume its training after it was interrupted, the
    History.history attribute contains the concatenation of the result of all the computations.
    """
    epochs = kwargs.get('epochs', 1)
    var_fname = f'{comp_dir}/{stem}_vars.pickle'
    model_fname = f'{comp_dir}/{stem}_model.h5'
    prev_history = None

    # Prepare the call-backs necessary to checkpoint the computation at the end of every epoch
    checkpoint_cb = CheckpointEpoch(comp_dir=comp_dir,  # TODO il looks like here prev_history is always None
                                    stem=stem,
                                    history=prev_history.history if prev_history is not None else None)
    callbacks = list(
        kwargs.get('callbacks', []))  # Workaround for an issue with hyperopt, that turns the list into a tuple
    callbacks.append(checkpoint_cb)
    kwargs['callbacks'] = callbacks

    # Look for a Keras ModelCheckpoint callback among the provided callbacks, if any
    model_checkpoint_cb = None
    for cb in callbacks:
        if isinstance(cb, tf.keras.callbacks.ModelCheckpoint):
            checkpoint_cb.model_checkpoint_cb = cb
            model_checkpoint_cb = cb
            break

    ''' Check if files are available with the state of a previously interrupted or completed computation; if so, load
        them '''
    if Path(var_fname).is_file():
        with open(var_fname, 'rb') as pickle_f:
            pickled = pickle.load(pickle_f)
        epoch = pickled['epoch']
        next_epoch = epoch[-1] + 1
        # model = tf.keras.models.load_model(model_fname)
        model, recompile = load_keras_model(model_fname, compile=False)
        compile_cb(model)
        prev_history = tf.keras.callbacks.History()
        prev_history.history = pickled['history']
        prev_history.model = model
        prev_history.epoch = epoch
        prev_history.params = pickled['params']
        # If the requested number of epochs has been completed already, then stop here returning the results
        if next_epoch >= epochs:
            return prev_history
        # Model.fit() will compute these many epochs: kwargs['epochs']-kwargs['initial_epoch']
        initial_epoch = kwargs.get('initial_epoch', epoch[0] if epoch else 0)
        kwargs['initial_epoch'] = max(initial_epoch, next_epoch)
        ''' If there is a Keras ModelCheckpoint callback among the callbacks, then update its state to resume the 
        computation from where it was interrupted '''
        if model_checkpoint_cb is not None:
            model_checkpoint_cb.epochs_since_last_save = pickled['epochs_since_last_save']
            model_checkpoint_cb.best = pickled['best']

    # Fit the model
    history = model.fit(**kwargs)

    # Concatenate the history with the history of previous computations (if any) and return the result
    if prev_history is not None:
        ''' Instantiate a new History object; avoid to modify the one returned by fit() as fit() returns a reference
        to one of the attributes of the model '''
        updated_history = tf.keras.callbacks.History()
        updated_history.epoch = prev_history.epoch + history.epoch
        for k, v in prev_history.history.items():
            updated_history.history[k] = prev_history.history[k] + history.history[k]
        history = updated_history

    return history


def make_logger(name, log_level):
    """
    Initializes and return a logger. See https://docs.python.org/3/library/logging.html
    :param name: the name for the logger, a string.
    :param log_level: the requested log level, as documented in https://docs.python.org/3/library/logging.html#levels
    :return: an instance of logging.Logger
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s: %(levelname)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class Trainer():
    def __init__(self, model, compile_cb, comp_dir, stem, space, val_metric, optimization_direction, log_dir=None,
                 make_train_dataset_cb=None, log_level=logging.INFO):
        '''
        An object that implements a callback for hp.tfmin(), and provides it with a state
        Be aware of https://github.com/tensorflow/tensorflow/issues/41021#issuecomment-786715361 TLDR: do not use
        the same instance of Optimizer to compile the model multiple times, otherwise TF may give an error
        when trying to save the model in a HDF5 (.h5) file.
        '''
        assert optimization_direction in ('min', 'max')

        self.logger = make_logger(name='Trainer', log_level=log_level)

        self.max_evals = None
        ''' Try to load the trials object from the file system (to resume an interrupted computation, or recover the 
        result of a completed one); if it fails, then instantiate a new one (start a new computation) '''
        self.trials_fname = f'{comp_dir}/{stem}_trials.pickle'
        if Path(self.trials_fname).is_file():
            with open(self.trials_fname, 'br') as trials_f:
                self.trials = pickle.load(trials_f)
        else:
            self.trials = Trials()
        self.model = model
        self.saved_model_fname = None
        self.compile_cb = compile_cb
        self.comp_dir = comp_dir
        self.stem = stem
        self.space = space
        self.trial = None
        self.log_dir = log_dir
        self.val_metric = val_metric
        self.optimization_direction = optimization_direction
        self.make_train_dataset_cb = make_train_dataset_cb
        self.report_file = f'{comp_dir}/{stem}.csv'
        self.report = None
        self.summary_report_file = f'{self.comp_dir}/{self.stem}_summary.csv'

    def make_trial_stem(self, stem, trial):
        trial_stem = '{}-{:04d}'.format(stem, trial)
        return trial_stem

    def resumable_fit_wrapper(self, params):
        self.trial += 1
        ''' If it is the first trial, checkpoint the model on disk before optimization starts. In subsequent trials,
        reload the checkpointed model from disk, to ensure every run of the optimization process starts from the same
        model and in the same state '''
        if self.trial == 0:
            self.logger.info(
                f'Starting trial 0, saving model {self.model.name} at beginning of trial in file {self.saved_model_fname}')
            self.compile_cb(self.model)
            self.saved_model_fname = f'{self.comp_dir}/{self.stem}_orig.h5'
            save_keras_model(self.model, filepath=self.saved_model_fname, save_format='h5')
        else:
            self.logger.info(f'Starting trial {self.trial}, reloading model from {self.saved_model_fname}')
            self.model, recompile = load_keras_model(self.saved_model_fname, compile=False)
            self.compile_cb(self.model)
        trial_stem = self.make_trial_stem(self.stem, self.trial)
        best_model_for_trial_fname = self.comp_dir + '/' + self.make_trial_stem(self.stem, self.trial) + '_best.h5'
        ''' Apply parameters from `params` to received callbacks as necessary. If there are Keras callbacks to 
        checkpoint the best model and/or save logs for Tensorboard, set the respective file names such that they 
        contain the current trial number. This will ensure a different file name for every trial.'''
        callbacks = params.get('callbacks')
        if callbacks is not None:
            for cb in callbacks:
                # File name for the checkpoint with the best validated model of the trial
                if isinstance(cb, tf.keras.callbacks.ModelCheckpoint):
                    cb.filepath = best_model_for_trial_fname
                # Directory name for logs for Tensorboard
                elif isinstance(cb, tf.keras.callbacks.TensorBoard):
                    assert (self.log_dir is not None)
                    cb.log_dir = self.log_dir + '/' + trial_stem
                # Parameters for EWA decaying learning rate
                elif isinstance(cb, EWA_LearningRateScheduler):
                    alpha = params.get('alpha')
                    if alpha is not None:
                        cb.alpha = alpha
                        del params['alpha']
                    decay = params.get('decay')
                    if decay is not None:
                        cb.decay = decay
                        del params['decay']
                    k = params.get('k')
                    if k is not None:
                        del params['k']
                        cb.k = k
        if self.make_train_dataset_cb is not None:
            x, params = self.make_train_dataset_cb(**params)
            if x is not None:
                assert params.get('x') is None
                params['x'] = x

        start_time = time()
        history = resumable_fit(model=self.model,
                                comp_dir=self.comp_dir,
                                stem=trial_stem,
                                compile_cb=self.compile_cb,
                                **params)
        end_time = time()

        ''' If this was the last trial, remove the file with the model checkpointed at the beginning of the first trial,
        it is not needed anymore '''
        if self.trial == self.max_evals - 1:
            self.logger.info(f'Last trial completed, removing temporary file {self.saved_model_fname}')
            Path(self.saved_model_fname).unlink(missing_ok=True)
            self.saved_model_fname = None

        loss = -max(history.history[self.val_metric]) if self.optimization_direction == 'max' \
            else min(history.history[self.val_metric])

        best_epoch = np.argmin(history.history[self.val_metric]) if self.optimization_direction == 'min' \
            else np.argmax(history.history[self.val_metric])
        record = {'trial': self.trial,
                  'running time (sec)': end_time - start_time,
                  'epochs run': len(history.epoch),
                  'best epoch': best_epoch,
                  'hyperopt loss': loss}
        for k, v in history.history.items():
            record[k] = v[best_epoch]

        record['file name'] = best_model_for_trial_fname

        for k, v in self.trials.trials[-1]['misc']['vals'].items():
            record[k] = v[0]
        if Path(self.report_file).is_file():
            self.report = pd.read_csv(self.report_file)
            self.report = self.report.append(pd.Series(record), ignore_index=True)
        else:
            self.report = pd.DataFrame.from_records([record])
        self.report.to_csv(self.report_file + '.tmp', index=False)
        keep_last_two_files(self.report_file)

        res = {'status': STATUS_OK,
               'loss': loss,
               'history': history.history,
               'model_file_name': best_model_for_trial_fname,
               'trial': self.trial}

        self.logger.info(
            f'Trial {self.trial} completed. Best model saved in {best_model_for_trial_fname} with hyperopt loss {loss}.')

        return res

    def do_it(self, max_evals, **kwargs):
        """
        Runs a given number of model optimizations, and optionally validations, using hyperopt to optimize the
        hyper-parameters.
        :param max_evals: the number of runs to be used to optimize hyper-parameters, corresponds to parameter max_evals
        of hyperopt.fmin().
        :param kwargs: parameters to be passed to hyperopt.fmin(). They are not allowed to include any of these:
        fn, space, max_evals, trials, trials_save_file; this is because the method needs to fill them itself.
        :return: the return value of hyperopt.fmin(), a dictionary with at least two keys 'status' and 'loss'.
        """
        self.logger.info(f'Requested to run {max_evals} trials, with trials state saved in {self.trials_fname}')
        for k, _ in kwargs.items():
            assert k not in ('fn', 'space', 'max_evals', 'trials', 'trials_save_file')
        self.trial = -1  # It gets incremented at the beginning of every trial, first trial will be number 0
        Path(self.comp_dir).mkdir(parents=True, exist_ok=True)
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.max_evals = max_evals
        start_time = time()
        best = fmin(fn=self.resumable_fit_wrapper,
                    space=self.space,
                    max_evals=max_evals,
                    trials=self.trials,
                    trials_save_file=self.trials_fname,
                    **kwargs)
        end_time = time()

        tr, nt, tot = count_weights(self.model)
        report = {'Running time (h:mm:s)': timedelta(seconds=end_time - start_time),
                  'Running time (min)': (end_time - start_time) / 60,
                  'Trials': len(self.trials.results),
                  'Best trial': self.trials.best_trial['tid'],
                  'Best trial hyperopt loss': self.trials.best_trial['result']['loss'],
                  'Best model saved in': self.trials.best_trial['result']['model_file_name'],
                  'Trials report file': self.report_file,
                  'Summary report': self.summary_report_file,
                  'Computation dir.': self.comp_dir,
                  'Log dir.': self.log_dir,
                  'Stem': self.stem,
                  'Model name': self.model.name,
                  'Trainable parameters': tr,
                  'Non-trainable parameters': nt,
                  'Total parameters': tot,
                  'Evaluation metric': self.val_metric,
                  'Optimization direction': self.optimization_direction}

        if Path(self.summary_report_file).is_file():
            self.logger.info(f'Summary file {self.summary_report_file} exists already, not overwriting it')
        else:
            pd.Series(report).to_csv(self.summary_report_file, index=True)
        self.logger.info("All trials completed")
        for k, v in report.items():
            self.logger.info(f'   {k}: {v}')

        Path(self.report_file + '.prev').unlink(missing_ok=True)
        return best


''' TODO
Add INFO logging messages to the resumable_fit()
Check that re-using callbacks between folds and/or trials doesn't mess them up
Integrate with the X-ray classification
Try fancy/cyclic learning rates
'''
