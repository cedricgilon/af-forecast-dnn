import os
import numpy as np
import pandas as pd
import forecast.hyperparameters as hp
from forecast.split import patient_split_by_nb_af
import util.path as mypath


def create_dataset_forecast(window, distance, tolerance, reshape=True):
    print(f"Load patient file")
    df = get_df()

    print(f"Create dataset w={window}, d={distance}, t={tolerance}")
    train_patient_id, test_patient_id, val_patient_id = patient_split_by_nb_af(df, window, distance, tolerance)

    train_df = df.loc[df['patient_id'].isin(train_patient_id)]
    test_df = df.loc[df['patient_id'].isin(test_patient_id)]
    val_df = df.loc[df['patient_id'].isin(val_patient_id)]

    print("-" * 50)
    print("Train")
    train_x, train_y = construct_dataset(train_df, window, tolerance, distance, reshape)
    print("-" * 50)
    print("Test")
    test_x, test_y = construct_dataset(test_df, window, tolerance, distance, reshape)
    print("-" * 50)
    print("Validation")
    val_x, val_y = construct_dataset(val_df, window, tolerance, distance, reshape)
    print("-" * 50)

    return train_x, train_y, test_x, test_y, val_x, val_y


def get_df():
    """
    Get a dataframe from a list of patient ids
    """
    li = []
    for i in range(0, hp.PATIENT_NUMBER):
        path = os.path.join(mypath.DATA_PATH, f"patient_{i}.csv")
        df_pat = pd.read_csv(path, index_col=None, header=0)
        li.append(df_pat)
    df = pd.concat(li, axis=0, ignore_index=True)
    return df


def construct_dataset(df, window, tolerance, distance, reshape):
    print("Search for AF")
    patient_id = df.patient_id.values
    af = df.AF.values
    rr = df.RR.values
    start_af = []

    for i in range(window+tolerance+distance, len(df)):
        if af[i - 1] == 0 and af[i] == 1 \
                and patient_id[i - window - tolerance - distance] == patient_id[i] \
                and not np.any(af[i - window - tolerance - distance:i]):
            start_af.append(i)

    print("Number of AF in the dataset:", len(start_af))
    x, y = add_af(start_af, af, rr, window, tolerance, distance)
    print("Size dataset with AF", len(x), len(y))
    x, y = add_not_af(x, y, patient_id, af, rr, window)
    print("Size dataset with AF and not AF", len(x), len(y))
    if reshape:
        x = np.reshape(x, (len(x), window, 1))
    return x, y


def add_af(start_af, af, rr, window, tolerance, distance):
    x = []
    y = []

    for i in start_af:
        for t in range(tolerance):
            if not np.any(af[i-window-t-distance:i]) and af[i] == 1:
                x.append(rr[i-window-t-distance:i-t-distance])
                y.append(1)
    return x, y


def add_not_af(x, y, patient_id, af, rr, window, distance_not_af=10000, size_not_af=1):
    number_af = len(x) * size_not_af

    while len(x) < (2 * number_af):
        random = np.random.randint(len(rr) - window - distance_not_af)
        if patient_id[random] == patient_id[random + window + distance_not_af] and \
                not np.any(af[random:random + window + distance_not_af]):
            x.append(rr[random:random + window])
            y.append(0)
    return x, y