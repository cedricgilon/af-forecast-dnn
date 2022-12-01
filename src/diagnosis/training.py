import os
import sys
import numpy as np
from datetime import datetime
from util.logger import Logger
from diagnosis.dataset import create_dataset_diagnosis
from diagnosis.model import get_model, get_callbacks
from util.metrics import metrics
from util.figures import graph_network_acc_loss, plot_roc, plot_cm
import diagnosis.hyperparameters as hp
import util.path as mypath
import pickle


def train_diagnosis():
    """
    Main function
    1) Create log
    2) Create model
    3) Train model
    4) Save model weights
    5) Test model
    6) Save figures
    """
    directory, timestamp = start_log()

    seed = np.random.randint(100)
    print(f"Seed: {seed}")
    np.random.seed(seed)

    train_x, train_y, test_x, test_y, val_x, val_y = create_dataset_diagnosis(window=hp.WINDOW,
                                                                              overlap=hp.OVERLAP)
    model = get_model("cnn_bigru")
    callbacks, histories = get_callbacks(hp.PATIENCE)

    # train model on data
    model.fit(x=train_x,
              y=train_y,
              batch_size=hp.BATCH_SIZE,
              epochs=hp.EPOCH,
              verbose=1,
              validation_data=(val_x, val_y),
              shuffle=True,
              callbacks=callbacks
              )

    model.save(directory + f"/model.h5")

    y_pred = model.predict(test_x)
    metrics(test_y, y_pred)

    plot_cm(test_y, y_pred, directory + "/cm.png")
    plot_roc(test_y, y_pred, "Diagnosis", directory + "/auc.png")

    graph_network_acc_loss(histories, directory + f"/histories.png")
    with open(directory + f"/histories.dump", "wb") as f:
        pickle.dump(histories, f)


def start_log():
    """
    Create log file and redirect output
    """
    timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
    directory = os.path.join(mypath.LOG_PATH, f"dia_w{hp.WINDOW}_o{hp.OVERLAP}/{timestamp}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    log_file = directory + f"/{timestamp}.log"
    sys.stdout = Logger(log_file)
    print(timestamp)
    print(log_file)
    print("window:", hp.WINDOW, "overlap:", hp.OVERLAP)
    print("epoch:", hp.EPOCH, "patience:", hp.PATIENCE, "lr:", hp.LEARNING_RATE)
    return directory, timestamp
