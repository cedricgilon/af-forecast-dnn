import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def metrics(y_test, y_pred, threshold=0.5, verbose=1):
    """
    Compute all the metrics for a given prediction
    """
    auc_fpr, auc_tpr, auc_threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(auc_fpr, auc_tpr)

    print("-" * 80)
    print(f"Threshold: {threshold}")
    y_pred_int = [1 if y_ >= threshold else 0 for y_ in y_pred]

    cm = confusion_matrix(y_test, y_pred_int)
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (recall * precision) / (recall + precision)

    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    false_positive = fp / total
    false_negative = fn / total

    if verbose:
        print('-' * 80)
        print(f'AUC:           {roc_auc}')
        print(f'F1:            {f1}')
        print(f'Precision:     {precision}')
        print(f'Recall:        {recall}')
        print('-' * 50)
        print(f'Accuracy:      {accuracy}')
        print(f'Sensitivity:   {sensitivity}')
        print(f'Specificity:   {specificity}')
        print(f'PPV:           {ppv}')
        print(f'NPV:           {npv}')
        print(f'False positive {false_positive}')
        print(f'False negative {false_negative}')
        print('-' * 80)
        print(cm)

    return roc_auc
