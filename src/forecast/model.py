from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GRU, Bidirectional, Conv1D, Reshape, Input, GlobalMaxPool1D
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
import keras.backend as K
import forecast.hyperparameters as hp


def get_model(window):
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=3, input_shape=(window, 1)))
    model.add(Conv1D(filters=100, kernel_size=3))
    model.add(GlobalMaxPool1D())
    model.add(Reshape((100, 1)))
    model.add(Bidirectional(GRU(units=100)))
    model.add(Dense(units=1,
                    activation="sigmoid"))

    o = optimizers.Adam(lr=hp.LEARNING_RATE)
    model.compile(optimizer=o,
                  loss=hp.LOSS_FUNCTION,
                  metrics=[f1, "acc"])
    model.summary(line_length=150)
    return model


def f1(y_true, y_pred):
    """
    Compute F1 score to be used as a metric
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def get_callbacks():
    """
    Create the callbacks for a given model
    """
    es = EarlyStopping(monitor='val_loss',
                       patience=hp.PATIENCE,
                       verbose=2,
                       mode='min',
                       restore_best_weights=True)
    histories = Histories()
    return [es, histories], histories


class Histories(Callback):
    """
    List of metrics and variables to be recorded during training
    """

    def __init__(self):
        super().__init__()
        self.accuracies = []
        self.losses = []
        self.f1 = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.f1.append(logs.get('f1'))