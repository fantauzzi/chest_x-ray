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
    res = []
    for layer in layers:
        if hasattr(layer, 'trainable'):
            res.append((layer.name, layer.trainable))
        if hasattr(layer, 'layers'):
            layer_res = collect_trainable_states(layer.layers)
            res.extend(layer_res)
    return res


def set_trainable_states(layers, names_and_state, idx=0):
    for layer in layers:
        if hasattr(layer, 'trainable'):
            name, layer.trainable = names_and_state[idx]
            assert layer.name == name
            idx = idx + 1
        if hasattr(layer, 'layers'):
            idx = set_trainable_states(layer.layers, names_and_state, idx)
    return idx


def count_components(variables):
    res = sum([np.prod(variable.shape) for variable in variables])
    return res


def count_weights(layer_or_model):
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
    pickle_fname = make_pickle_file_name(filepath)
    trainable = collect_trainable_states(model.layers)
    tf.keras.models.save_model(model, filepath, overwrite, include_optimizer, save_format, signatures, options,
                               save_traces)
    with open(pickle_fname, 'wb') as pickle_f:
        pickle.dump(trainable, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)


def load_keras_model(filepath, custom_objects=None, compile=True, options=None):
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

    def save_checkpoint(self, epoch, logs):
        # Pickle variables that are needed to restore the computation but are not saved in a .h5 file
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
        if self.model.history.epoch:
            # Save a checkpoint with the previous epoch (the last epoch that completed)
            self.save_checkpoint(epoch=epoch - 1, logs=logs)

    def on_train_end(self, logs=None):
        # Save a checkpoint with the last epoch
        self.save_checkpoint(epoch=len(self.model.history.epoch) + self.epochs_in_history - 1, logs=logs)
        # Remove the .prev files of the computation, not needed anymore
        Path(self.prev_model_fname).unlink(missing_ok=True)
        pickle_filepath = make_pickle_file_name(self.prev_model_fname)
        Path(pickle_filepath).unlink(missing_ok=True)
        Path(self.prev_vars_fname).unlink(missing_ok=True)


def make_k_fold_file_name(stem):
    fname = stem + '_kfold.pickle'
    return fname


def k_fold_resumable_fit(model, comp_dir, stem, compile_cb, make_datasets_cb, n_folds, log_level=logging.INFO,
                         log_dir=None, **kwargs):
    """

    :param model:
    :param comp_dir:
    :param stem:
    :param compile_cb:
    :param make_datasets_cb: a function or method that instantiates the training and, optionally, validation pipelines.
    It takes three parameters: the fold number, a non-negative integer, the total number of folds, a positive
    integer, and **kwargs. It returns two instances of td.data.Dataset, with the training and validation datasets
    respectively for the given fold number, or one instance and None, if no validation is required.
    :param n_folds:
    :param kwargs:
    :return:
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

    for fold in range(current_fold, n_folds):
        train_ds, val_ds = make_datasets_cb(fold, n_folds, **kwargs)
        kwargs['x'] = train_ds
        kwargs['validation_data'] = val_ds
        fold_stem = '{}-fold{:02d}'.format(stem, fold)
        logger.info(f'Processing fold {fold} - Total folds to be processed: {n_folds}')
        callbacks = kwargs.get('callbacks', [])
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.TensorBoard) and log_dir is not  None:
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
    return histories


def resumable_fit(model, comp_dir, stem, compile_cb, **kwargs):
    '''
    Trains, and optionally evaluate, a model, saving the training state at the end of every epoch; can resume a
    previously interrupted training from the end of the last epoch that was completed.
    :param model: the model to be trained, a Keras model. If a computation has been previously checkpointed,
    then the model will be loaded from the checkpoints, as needed, and this parameter is ignored, can be set to None.
    Otherwise, the model has to have been compiled already.
    :param comp_dir: path to the directory where the training state is checkpointed, and where the trained model is
    saved, a string.
    :param stem: a stem that will be used to generate file names to checkpoint the computation and save the results.
    :param compile_cb: a function that will be called to rebuild the Keras model as necessary. It must take one
    argument, which is the model to be compiled; anything it returns is ignored. It must be provided when the model
    has any layer set to non-trainable; otherwise it can be None.
    :param kwargs: keyword arguments to be passed to tf.Keras.Model.fit().
    :return: a History object with a record of the computation, as returned by tf.Keras.Model.fit(). In case of
    multiple calls to this function, to resume training after it has been interrupted, the History.history attribute
    contains the concatenation of the result of all the computations.
    '''
    epochs = kwargs.get('epochs', 1)
    var_fname = f'{comp_dir}/{stem}_vars.pickle'
    model_fname = f'{comp_dir}/{stem}_model.h5'
    prev_history = None

    # Prepare the call-backs necessary to checkpoint the computation at the end of every epoch
    checkpoint_cb = CheckpointEpoch(comp_dir=comp_dir,
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
        Careful with https://github.com/tensorflow/tensorflow/issues/41021
        '''
        assert optimization_direction in ('min', 'max')

        """self.logger = logging.getLogger('Trainer')
        self.logger.handlers = []
        self.logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s: %(levelname)s %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)"""
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
        ''' If there are Keras callbacks to checkpoint the best model and/or save logs for Tensorboard, set the 
        respective file names such that they contain the current trial number. This will ensure a different file
        name for every trial.'''
        ''' Apply parameters from `params` to received callbacks as necessary '''
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
        '''
        Runs a given number of optimizations, and optionally validations, optimizing the hyper-parameters.
        :param max_evals: the number of runs to be used to optimize hyper-parameters, corresponds to parameter max_evals
        of hyperopt.fmin().
        :param kwargs: parameters to be passed to hyperopt.fmin(). They are not allowed to include any of these:
        fn, space, max_evals, trials, trials_save_file.
        :return: the return value of hyperopt.fmin(), a dictionary with at least two keys 'status' and 'loss'.
        '''
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
Check that re-using callbacks between folds and/or trials doesn't mess them up
re-train the final model and test it
correct log dir for k-fold x-validation
charts and statistics for k-fold x-validation, choice of the final model for inference (and test)
Integrate with the X-ray classification
Document properly
Try fancy/cyclic learning rates
'''
