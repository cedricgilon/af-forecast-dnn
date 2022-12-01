import time
import os
import sys
from datetime import datetime
from util.logger import Logger
from forecast.dataset import create_dataset_forecast
from forecast.model import get_model, get_callbacks
from util.metrics import metrics
import numpy as np
import pickle
import forecast.hyperparameters as hp
import util.path as mypath


def train_forecast():
    start_time = time.time()

    directory, timestamp = start_log()

    seed = np.random.randint(hp.MAX_SEED)
    print(f"Seed: {seed}")
    np.random.seed(seed)

    dataset_time = time.time()
    train_x, train_y, test_x, test_y, val_x, val_y = create_dataset_forecast(window=hp.WINDOW,
                                                                             distance=hp.DISTANCE,
                                                                             tolerance=hp.TOLERANCE)

    train_x = train_x[:, -hp.WINDOW:]
    test_x = test_x[:, -hp.WINDOW:]
    val_x = val_x[:, -hp.WINDOW:]

    dataset_time = time.time() - dataset_time

    model = train(train_x, train_y, val_x, val_y)
    auc = test(model, test_x, test_y, timestamp, directory)
    model.save(directory + f"/{timestamp}-model.h5")

    run_time = time.time() - start_time
    print("--- %s seconds ---" % run_time)

    with open(directory + f"/{timestamp}_result_values", "wb") as f:
        a = [auc, run_time, dataset_time]
        pickle.dump(a, f)


def start_log():
    timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
    directory = os.path.join(mypath.LOG_PATH, f"for_w{hp.WINDOW}_d{hp.DISTANCE}_t{hp.TOLERANCE}/{timestamp}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = directory + f"/{timestamp}.log"
    sys.stdout = Logger(log_file)
    print(f"timestamp {timestamp}")
    print(f"log_file{log_file}")
    print(f"window {hp.WINDOW}, distance {hp.DISTANCE}, tolerance {hp.TOLERANCE}")
    print(f"epoch {hp.EPOCH}, patience {hp.PATIENCE}, lr {hp.LEARNING_RATE}")

    return directory, timestamp


def train(train_x, train_y, val_x, val_y):
    model = get_model(hp.WINDOW)
    callbacks, histories = get_callbacks()

    # train model on data
    model.fit(x=train_x,
              y=train_y,
              batch_size=hp.BATCH_SIZE,
              epochs=hp.EPOCH,
              verbose=2,
              validation_data=(val_x, val_y),
              shuffle=True,
              callbacks=callbacks
              )

    return model


def test(model, x_test, y_test, timestamp, directory):
    y_pred = model.predict(x_test)
    with open(directory + f"/{timestamp}_pred_test_d{hp.DISTANCE}_w{hp.WINDOW}", "wb") as f:
        a = [y_test, y_pred]
        pickle.dump(a, f)
    auc = metrics(y_test, y_pred)
    return auc
