from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import torch

def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)

def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.f1_score(y_true, y_pred, zero_division=1,average='binary')

def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.recall_score(y_true, y_pred, zero_division=1, average='binary')

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='binary')

def PPV(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='binary')

def NPV(output, target):
    conf_matrix = confusion_matrix(output, target)
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]  
    NPV = TN / (TN + FN)
    return NPV

def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4)



def spe(output, target):
    spe = []
    con_mat = confusion_matrix(output, target)
    for i in range(2):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return spe[1]

def roc(output, target, results_dir, file_name):

    y_true = target
    output = torch.tensor(output)
    probabilities = F.softmax(output, dim=1)
    values = []
    values0 = []

    for i in range(len(y_true)):
        selected_value = probabilities[i][1]
        values.append(selected_value.item())
        selected_value0 = probabilities[i][0]
        values0.append(selected_value0.item())
    y_pred = values
    y_pred0 = values0


    if len(np.unique(y_true)) == 1:
        auc = 0.5 
    else:
        auc = metrics.roc_auc_score(y_true, y_pred)
        auc0 = metrics.roc_auc_score(y_true, y_pred0)
    
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.title(file_name+'auc{:.2f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='AUC=%.4f' % auc)
    # file_name = 'roc_1.png'
    plt.savefig(os.path.join(results_dir, file_name+'auc{:.2f}.jpg'.format(auc)))
    print('auc={}'.format(auc))
    print('auc0={}'.format(auc0))
    return auc, y_true, y_pred
