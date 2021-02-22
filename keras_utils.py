import tensorflow as tf
from pathlib import Path
import pickle
import shutil


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
                 best_so_far,
                 best_epoch,
                 epoch_history,
                 history_history):
        super(CheckpointEpoch, self).__init__()
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
        if (already_computed_epochs >= max_epochs) or (best_epoch is not None and already_computed_epochs - best_epoch > patience):
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

    checkpoint_epoch_cb = CheckpointEpoch(file_name_stem=path_and_fname_stem,
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


