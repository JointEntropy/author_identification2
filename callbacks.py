from  keras.callbacks import Callback
from utils import DriveWrapper

from scoring import f2_grouped_score


class CompositionAveragePred(Callback):
    def __init__(self, val_data, val_target, groups_labels, ohe, val_part=1, branches=False):
        self.split_idx = int(val_target.shape[0] * val_part)
        self.val_data = val_data  # preprocessed validation data with no index, but shuffled as comp_groups val
        self.val_target = val_target
        self.groups_labels = groups_labels
        self.ohe = ohe
        self.branches = branches

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = dict()

        print('Epoch {} Scoring on first {} validation samples.'.format(epoch, self.split_idx))
        # отрезаем кусок на валидацию, потому что валидироваться на всём слишком долго
        if self.branches:
            val_data = [self.val_data[0][:self.split_idx]   , self.val_data[1][:self.split_idx]]
        else:
            val_data = self.val_data[0][:self.split_idx]
        # предсказываем метки
        pred = self.model.predict(val_data)
        score = f2_grouped_score(pred,
                          self.val_target[:self.split_idx],
                          self.groups_labels[:self.split_idx],
                          self.ohe)
        print('f2-average classes(median): {:.4f}'.format(score))


class GoogleDriveLogger(Callback):
    def __init__(self, save_name_frmt, callback_model_name='lstm.h5', on_each=3, init_epoch=0):
        self.save_name_frmt = save_name_frmt
        self.on_each = on_each
        self.callback_model_name = callback_model_name
        self.init_epoch = init_epoch
        if self.init_epoch !=0:
            print('Continue training from {} epoch!'.format(self.init_epoch))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = dict()
        epoch += self.init_epoch
        if epoch != 0 and epoch % self.on_each == 0:
            print('Saving best model to google drive.')
            gdw = DriveWrapper()
            gdw.save(self.callback_model_name, self.save_name_frmt.format(epoch))
