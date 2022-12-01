from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_cm(y_real, y_pred, filename):
    """
    Plots the confusion matrix
    input:
        - y_real: a numpy array containing the real results (1/0)
        - y_pred: a numpy array containing the predicted result (1/0)
    """
    plt.close()
    threshold = 0.5
    y_pred_int = [0 if y_ < threshold else 1 for y_ in y_pred]

    cm = confusion_matrix(y_real, y_pred_int)
    class_names = ['Not AF', 'AF']
    ax = sn.heatmap(cm, annot=True, fmt="d",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Number of samples'},
                    cmap=plt.cm.Blues)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label', rotation="horizontal", labelpad=30)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc(y_real, y_pred_proba, legend, filename):
    """
    Plots the ROC curve
    input:
        - y_real: a numpy array containing the real result (1/0)
        - y_pred_proba: a numpy array containing the predicted probabilities (0.0 - 1.0)
        - title: the title to give to the ROC curve
    """
    plt.close()
    plt.figure(figsize=(7, 7))
    fpr, tpr, _ = roc_curve(y_real, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    lw = 2
    str_auc = str("%.4f" % roc_auc)
    plt.plot(fpr, tpr, lw=lw, label=f"{legend} (AUC={str_auc})")
    plt.plot([-2, 2], [-2, 2], color='black', lw=lw, linestyle=':')
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()


def graph_network_acc_loss(histories, filename):
    """
    Plot the evolution graphes of accuracy, F1 score and loss
    """
    plt.close()
    plt.figure(figsize=(10, 5))
    title = 'Loss and accuracy'
    plt.title(title)

    plt.subplot(311)
    plt.ylabel('Loss')
    plt.plot(histories.losses)

    plt.subplot(312)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.0])
    plt.plot(histories.accuracies)

    plt.subplot(313)
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.plot(histories.f1)

    plt.savefig(filename)
    plt.close()
