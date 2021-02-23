import tensorflow as tf
from pathlib import Path
import pickle
import shutil
from copy import deepcopy


def keep_last_two_files(file_name):
    if Path(file_name).is_file():
        Path(file_name).replace(file_name + '.prev')
    Path(file_name + '.tmp').replace(file_name)


class CheckpointAndSaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self,
                 file_name_stem,
                 metric_key,
                 optimization_direction,
                 patience,
                 already_computed_epochs,
                 max_epochs,
                 best_so_far,
                 best_epoch,
                 epoch_history,
                 history_history):
        super(CheckpointAndSaveBestModel, self).__init__()
        assert optimization_direction in ['min', 'max']
        assert already_computed_epochs < max_epochs
        self.optimization_direction = optimization_direction
        self.metric_key = metric_key
        self.patience = patience
        self.best_so_far = best_so_far if best_so_far is not None else \
            float('inf') if optimization_direction == 'min' else float('-inf')
        self.best_epoch = best_epoch
        self.model_file_name = None
        self.file_name_stem = file_name_stem
        self.max_epochs = max_epochs
        self.history_epoch = epoch_history
        self.history_history = history_history

    def on_epoch_end(self, epoch, logs=None):
        # Save the model at the end of the epoch, to be able to resume training from there
        tf.keras.models.save_model(model=self.model,
                                   filepath=self.file_name_stem + '.h5.tmp',
                                   save_format='h5')
        keep_last_two_files(self.file_name_stem + '.h5')

        ''' If the last epoch had the best validation so far, then copy the saved model in a dedicated file,
        that can be loaded and used for testing and inference '''
        if self.metric_key is not None:
            metric_value = logs[self.metric_key]
            if (self.optimization_direction == 'max' and metric_value > self.best_so_far) or \
                    (self.optimization_direction == 'min' and metric_value < self.best_so_far):
                self.best_so_far = metric_value
                self.best_epoch = epoch
                new_model_file_name = self.file_name_stem + '_best.h5'
                print(
                    f'Best epoch so far {self.best_epoch} with {self.optimization_direction} {self.metric_key} = {self.best_so_far} -Saving model in file {new_model_file_name}')
                shutil.copyfile(self.file_name_stem + '.h5', self.file_name_stem + '_best.h5')

        # Save those variables, needed to resume training from the last epoch, that are not saved with the model
        for k, v in logs.items():
            self.history_history.setdefault(k, []).append(v)
        self.history_epoch = self.history_epoch + [epoch]
        pickle_this = {'epochs_total': epoch + 1,  # Epochs are numbered from 0
                       'best_so_far': self.best_so_far,
                       'best_epoch': self.best_epoch,
                       'epoch_history': self.history_epoch,
                       'history_history': self.history_history}
        pickle_fname = self.file_name_stem + '.pickle'
        with open(pickle_fname + '.tmp', 'bw') as pickle_f:
            pickle.dump(pickle_this, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        keep_last_two_files(pickle_fname)

        # Test the conditions to stop training, i.e. the max number of epochs has been exceeded, or early termination
        if epoch + 1 >= self.max_epochs:  # epochs are numbered from 0
            print(f'\nStopping training as it has reached the maximum number of epochs {self.max_epochs}')
            self.model.stop_training = True
        elif self.best_epoch is not None and self.patience is not None and epoch - self.best_epoch > self.patience:
            if self.patience == 0:
                print('\nStopping training has there has been no improvement since the previous epoch.')
            else:
                print(f'\nStopping training as there have been no improvements for more than {self.patience} epoch(s).')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        ''' Remove files with intermediate checkpoints of the optimization just completed. Keep only the .h5 file with
        the best validated model '''
        # If not required to save the best model, as no evaluation metric was provided, then save the last epoch model
        if self.metric_key is None:
            shutil.copyfile(self.file_name_stem + '.h5', self.file_name_stem + '_best.h5')
        Path(self.file_name_stem + '.h5').unlink(missing_ok=True)
        Path(self.file_name_stem + '.h5.prev').unlink(missing_ok=True)
        Path(self.file_name_stem + '.pickle.prev').unlink(missing_ok=True)


class LearningRateScheduler():
    def __init__(self, alpha, decay, k):
        self.alpha = alpha
        self.decay = decay
        self.k = k

    def update(self, epoch, lr):
        updated_lr = self.alpha * (self.decay ** (epoch / self.k))
        return updated_lr


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

    def on_train_begin(self, logs=None):
        Path(self.comp_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch, logs):
        trainable = [layer.trainable for layer in self.model.layers]
        updated_history = {}
        for k, v in self.model.history.history.items():
            updated_history[k] = self.history.get(k, []) + self.model.history.history[k]
        pickle_this = {'completed_epochs': epoch + 1,
                       'history': updated_history,
                       'params': self.model.history.params,
                       'trainable': trainable}
        with open(self.tmp_vars_fname, 'bw') as pickle_f:
            pickle.dump(pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        keep_last_two_files(self.vars_fname)
        # Note: save_format='tf' would be much slower
        tf.keras.models.save_model(self.model, filepath=self.tmp_model_fname, save_format='h5')
        keep_last_two_files(self.model_fname)

    def on_epoch_begin(self, epoch, logs=None):
        if self.model.history.epoch:
            self.save_checkpoint(epoch=epoch - 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        pass
        """print('\nOn epoch end', epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        trainable = [layer.trainable for layer in self.model.layers]
        pickle_this = {'completed_epochs': epoch + 1,
                       'history': self.history,
                       'params': self.model.history.params,
                       'trainable': trainable}
        with open(self.tmp_vars_fname, 'bw') as pickle_f:
            pickle.dump(pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        keep_last_two_files(self.vars_fname)
        # Note: save_format='tf' would be much slower
        tf.keras.models.save_model(self.model, filepath=self.tmp_model_fname, save_format='h5')
        keep_last_two_files(self.model_fname)"""

    def on_train_end(self, logs=None):
        # Save a checkpoint with the last epoch
        self.save_checkpoint(epoch=len(self.model.history.epoch) + self.epochs_in_history - 1, logs=logs)
        # Remove the .prev files of the computation, not needed anymore
        Path(self.prev_model_fname).unlink(missing_ok=True)
        Path(self.prev_vars_fname).unlink(missing_ok=True)


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
    :param kwargs: arguments to be passed to tf.Keras.Model.fit(). They are not allowed to include 'initial_epoch'.
    :return: a History object with a record of the computation, as returned by tf.Keras.Model.fit(). In case of
    multiple calls to this function, to resume training after it has been interrupted, the History.history attribute
    contains the concatenation of the result of all the computations.
    '''
    ''' Check if files are available with the state of a previously interrupted or completed computation; if so, load 
    them '''
    assert kwargs.get('initial_epoch') is None
    epochs = kwargs.get('epochs', 1)
    var_fname = f'{comp_dir}/{stem}_vars.pickle'
    model_fname = f'{comp_dir}/{stem}_model.h5'
    prev_history = None
    if Path(var_fname).is_file():
        with open(var_fname, 'rb') as pickle_f:
            pickled = pickle.load(pickle_f)
        # TODO read the pickled variables here
        completed_epochs = pickled['completed_epochs']
        model = tf.keras.models.load_model(model_fname)
        trainable = pickled['trainable']
        assert len(trainable) == len(model.layers)
        if sum(trainable) < len(trainable):
            assert (compile_cb is not None)
            for layer, is_trainable in zip(model.layers, trainable):
                layer.trainable = is_trainable
            compile_cb(model)
        prev_history = tf.keras.callbacks.History()
        prev_history.history = pickled['history']
        prev_history.model = model
        prev_history.epoch = list(range(completed_epochs))
        prev_history.params = pickled['params']
        # If the requested number of epochs has been completed already, then stop here returning the results
        if completed_epochs >= epochs:
            return prev_history
        # Model.fit() will compute these many epochs: epochs-initial_epoch
        kwargs['initial_epoch'] = completed_epochs

    # Prepare the call-backs necessary to checkpoint the computation at the end of every epoch
    checkpoint_cb = CheckpointEpoch(comp_dir=comp_dir,
                                    stem=stem,
                                    history=prev_history.history if prev_history is not None else None)
    callbacks = kwargs.get('callbacks', [])
    callbacks.append(checkpoint_cb)
    kwargs['callbacks'] = callbacks

    # Fit the model
    history = model.fit(**kwargs)
    # Concatenate the history with the history of previous computations (if any) and return the result
    if prev_history is not None:
        ''' Instantiate a new History object, avoid to modify the one returned by fit() as fit() returns a reference
        to one of the attributes of the model '''
        updated_history = tf.keras.callbacks.History()
        updated_history.epoch = prev_history.epoch + history.epoch
        for k, v in prev_history.history.items():
            updated_history.history[k] = prev_history.history[k] + history.history[k]
        history = updated_history

    return history


class Trainer():
    def __init__(self, comp_dir, max_evals):
        '''
        An object that implements a callback for hp.tfmin(), and provides it with a state
        '''
        self.max_evals = max_evals
        ''' Try to load the trials object from the file system (to resume an interrupted computation, or recover the 
        result of a completed one); if it fails, then instantiate a new one (start a new computation) '''
        self.trials = None

    def do_it(**params):
        '''
        The callback to be passed to hp.fmin() as its argument fn
        :param params: the parameters that the method receives from fmin()
        :return: the return value requested by hp.fmin(), a dictionary with at least to keys 'status' and 'loss'
        '''
        pass


def checkpointed_fit(model,
                     path_and_fname_stem,
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
    epoch_history = []
    history_history = {}

    # Load a previously saved model, if it exists, and the corresponding pickled variables
    pickled_fname = path_and_fname_stem + '.pickle'
    if Path(pickled_fname).is_file():
        with open(path_and_fname_stem + '.pickle', 'br') as pickle_f:
            pickled = pickle.load(pickle_f)
        # Total number of epochs the model has already been optimized for (>=1)
        already_computed_epochs = pickled['epochs_total']
        # Best value of the validation metric during the optimization so far
        best_so_far = pickled['best_so_far']
        # Epoch when the best value of the validation metric was obtained (epochs are numbered starting from 0)
        best_epoch = pickled['best_epoch']
        epoch_history = pickled['epoch_history']
        history_history = pickled['history_history']

        # Check if the model has already met the criteria to stop training, and in case do not resume training
        if (already_computed_epochs >= max_epochs) or (
                best_epoch is not None and already_computed_epochs - best_epoch > patience):
            if already_computed_epochs >= max_epochs:
                print(f'\nThe model has already been trained for the maximum number of epochs {max_epochs}.')
            elif already_computed_epochs - best_epoch > patience:
                print(
                    f'\nModel training already stopped after {already_computed_epochs} epoch(s) because it exceeded {patience} epoch(s) without improvement on metric {metric_key}.')
            else:
                assert False
            history = tf.keras.callbacks.History()
            history.history = history_history
            history.model = model
            return history
        checkpoint_fname = path_and_fname_stem + '.h5'
        print(f'\nResuming optimization from the model loaded from {checkpoint_fname}')
        model = tf.keras.models.load_model(checkpoint_fname)

        print(
            f'\nResuming training after epoch {already_computed_epochs}. So far the best model validation had {metric_key}={best_so_far} at epoch {best_epoch}.')

    checkpoint_epoch_cb = CheckpointAndSaveBestModel(file_name_stem=path_and_fname_stem,
                                                     metric_key=metric_key,
                                                     optimization_direction=optimization_direction,
                                                     patience=patience,
                                                     already_computed_epochs=already_computed_epochs,
                                                     max_epochs=max_epochs,
                                                     best_so_far=best_so_far,
                                                     best_epoch=best_epoch,
                                                     epoch_history=epoch_history,
                                                     history_history=history_history)

    scheduler = LearningRateScheduler(alpha=alpha, decay=decay, k=k)
    learning_rate_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler.update, verbose=1)
    callbacks = args.get('callbacks', [])
    callbacks += [checkpoint_epoch_cb, learning_rate_scheduler_cb]
    args['callbacks'] = callbacks
    history = model.fit(initial_epoch=already_computed_epochs, **args)
    history.epoch = checkpoint_epoch_cb.history_epoch
    history.history = checkpoint_epoch_cb.history_history
    if history.model is None:
        history.model = model
    return history
