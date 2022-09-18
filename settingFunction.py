import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

auc_val_value, recall_val_value, prec_val_value, f1_val_value, cm_val_value = [], [], [], [], []

def DataImport(data):

    X = data.drop('target', axis=1)  # 刪除target列之欄位
    y = data['target']
    '''
    #剔除射頻功率
    X = X.drop('p1', axis=1)
    X = X.drop('p2', axis=1)
    X = X.drop('p3', axis=1)
    X = X.drop('p4', axis=1)
    '''
    variables = X.values
    type_label = (data['target']).values
    X_training, X_testing, y_training, y_testing = train_test_split(variables, type_label, test_size=0.2,
                                                                    random_state=1)  # 80% training and 20% test
    return X, y, X_training, X_testing, y_training, y_testing


def ScoreReport(y_t, predictions_t):
    accuracy = sklearn.metrics.accuracy_score(y_t, predictions_t)
    recall = sklearn.metrics.recall_score(y_t, predictions_t)
    precision = sklearn.metrics.precision_score(y_t, predictions_t)
    f1 = sklearn.metrics.f1_score(y_t, predictions_t)
    cm = confusion_matrix(y_t, predictions_t)
    print(cm)
    t_scores = [accuracy, recall, precision, f1]
    return t_scores

def roc(title,model, X_testing, y_testing):
    # fpr, tpr, thresholds = roc_curve(y_t, predictions_t)
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (title, roc_auc))
    y_score = model.decision_function(X_testing)
    fpr, tpr, threshold = roc_curve(y_testing, y_score)  ###計算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###計算auc的值
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (title, roc_auc))  ###假正率為橫座標，真正率為縱座標做曲線

def rocshow(title):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()