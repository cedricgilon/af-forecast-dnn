import math
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import diagnosis.hyperparameters as hp


def create_dataset_diagnosis(window, overlap, reshape=True):
    """
    A function to create lists containing our diagnosis objective, based on the raw patients data
    Returns:
        train_x and train_y
        test_x and test_y
        val_x and val_y
    """
    print(f"Create dataset for diagnosis (window={window},overlap={overlap})")

    patient_id = range(hp.PATIENT_NUMBER)
    train_patient_id, test_patient_id = train_test_split(patient_id, test_size=0.1, shuffle=True)
    split_size = math.floor((0.1 / (1 - 0.1)) * 100) / 100
    train_patient_id, val_patient_id = train_test_split(train_patient_id, test_size=split_size, shuffle=True)
    print(f"Train {len(train_patient_id)}, test {len(test_patient_id)}, val {len(val_patient_id)}")
    print("Train", train_patient_id)
    print("Test", test_patient_id)
    print("Val", val_patient_id)

    train_df = get_df_from_id(train_patient_id)
    test_df = get_df_from_id(test_patient_id)
    val_df = get_df_from_id(val_patient_id)

    train_x, train_y = create_x_y(train_df, window, overlap, reshape)
    test_x, test_y = create_x_y(test_df, window, overlap, reshape)
    val_x, val_y = create_x_y(val_df, window, overlap, reshape)

    return train_x, train_y, test_x, test_y, val_x, val_y


def get_df_from_id(list_id):
    """
    Get a dataframe from a list of patient ids
    """
    li = []
    for i in list_id:
        path = os.path.join(hp.DATA_PATH, f"patient_{i}.csv")
        df_pat = pd.read_csv(path, index_col=None, header=0)
        li.append(df_pat)
    df = pd.concat(li, axis=0, ignore_index=True)
    return df


def create_x_y(df, window, overlap, reshape):
    """
    Create x and y list (data and corresponding label)
    """
    print(len(df))
    patient_id = df.patient_id.values
    rr = df.RR.values
    af = df.AF.values
    x = []
    y = []
    step = window - overlap
    for i in tqdm(range(0, len(df) - window, step)):
        if patient_id[i] == patient_id[i + window]:
            x.append(rr[i:i+window])
            y.append(1 if np.any(af[i:i + window]) else 0)
    if reshape:
        x = np.reshape(x, (len(x), window, 1))
    print(len(x), len(y))
    return x, y
