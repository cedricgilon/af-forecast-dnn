from sklearn.utils import shuffle
import numpy as np


def patient_split_by_nb_af(df, window, distance, tolerance, train_split=0.8, test_split=0.1, val_split=0.1):
    if (train_split + test_split + val_split) != 1:
        raise ValueError("Split is not correct (sum != 1)")

    d = create_nb_af_dict(df, window, distance, tolerance)

    while True:
        patient, nb_af = shuffle(list(d.keys()), list(d.values()))
        total = sum(d.values())
        train_index, test_index = find_train_test_index(nb_af, total, train_split, test_split)

        train_nb_af = nb_af[:train_index]
        test_nb_af = nb_af[train_index:test_index]
        val_nb_af = nb_af[test_index:]

        # print(abs((sum(train_nb_af)/total) - train_split))
        # print(abs((sum(test_nb_af)/total) - test_split))
        # print(abs((sum(val_nb_af)/total) - val_split))

        if abs((sum(train_nb_af)/total) - train_split) < 0.02 \
                and abs((sum(test_nb_af)/total) - test_split) < 0.01\
                and abs((sum(val_nb_af)/total) - val_split) < 0.01:
            break

    train_patient = patient[:train_index]
    test_patient = patient[train_index:test_index]
    val_patient = patient[test_index:]
    print("Train patient\n", train_patient)
    print("Test patient\n", test_patient)
    print("Val patient\n", val_patient)

    print(f"Number AF: train {sum(train_nb_af)}, test {sum(test_nb_af)}, val {sum(val_nb_af)}, total {total}")
    print(f"Percentage number AF train {sum(train_nb_af)/total}")
    print(f"Percentage number AF test {sum(test_nb_af) / total}")
    print(f"Percentage number AF val {sum(val_nb_af) / total}")

    return train_patient, test_patient, val_patient


def create_nb_af_dict(df, window, distance, tolerance):
    af = df.AF.values
    patient_id = df.patient_id.values
    rr = df.RR.values
    d = dict()
    for i in range(window+distance+tolerance, len(rr)):
        if af[i - 1] == 0 and af[i] == 1 \
                and patient_id[i - window - tolerance - distance] == patient_id[i] \
                and not np.any(af[i - window - tolerance - distance:i]):
            if patient_id[i] not in d:
                d[patient_id[i]] = 1
            else:
                d[patient_id[i]] = d[patient_id[i]] + 1
    return d


def find_train_test_index(nb_af, total, train_split, test_split):

    nb_af_sum = 0
    train_index = -1
    test_index = -1
    for i in range(len(nb_af)):
        if train_index == -1:
            if nb_af_sum < (total * train_split):
                nb_af_sum += nb_af[i]
            else:
                train_index = i
        elif test_index == -1:
            if nb_af_sum < (total * (train_split + test_split)):
                nb_af_sum += nb_af[i]
            else:
                test_index = i
        else:
            break
    return train_index, test_index
