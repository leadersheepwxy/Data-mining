import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

auc_val_value, recall_val_value, prec_val_value, f1_val_value, cm_val_value = [], [], [], [], []


def DataImport(data):
    X = data.drop('income', axis=1)  # 刪除target列之欄位
    y = data['income']
    variables = X.values
    type_label = (data['income']).values
    X_training, X_testing, y_training, y_testing = train_test_split(variables, type_label, test_size=0.2, random_state=1)  # 80% training and 20% test
    return X, y, X_training, X_testing, y_training, y_testing


def ScoreChart(mode, accuracy, recall, precision, f1):
    mode = str(mode)
    t_scores = [accuracy, recall, precision, f1]
    t_scores_name = ['accuracy', 'recall', 'precision', 'f1']
    t_scores_chart = pd.DataFrame(t_scores, t_scores_name)
    if 'y' in mode:
        print(t_scores_chart)


def ScoreReport(y_t, predictions_t):
    accuracy = sklearn.metrics.accuracy_score(y_t, predictions_t)
    recall = sklearn.metrics.recall_score(y_t, predictions_t)
    precision = sklearn.metrics.precision_score(y_t, predictions_t)
    f1 = sklearn.metrics.f1_score(y_t, predictions_t)
    cm = confusion_matrix(y_t, predictions_t)
    print(cm)
    t_scores = [accuracy, recall, precision, f1]
    return t_scores


def ScoreList(mode, y_t, predictions_t):
    mode = str(mode)
    accuracy, recall, precision, f1, cm = ScoreReport(mode, y_t, predictions_t)
    auc_val_value.append(accuracy)
    recall_val_value.append(recall)
    prec_val_value.append(precision)
    f1_val_value.append(f1)
    cm_val_value.append(cm)
    return auc_val_value, recall_val_value, prec_val_value, f1_val_value, cm_val_value


def roc(title, y_t, predictions_t):
    fpr, tpr, thresholds = roc_curve(y_t, predictions_t)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (title, roc_auc))
    #plt.plot([0, 1], [0, 1], color='green', linestyle='--')

def rocshow(title):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def lowerSample(df,percent=1):
    data_more = df[df['income']==0]  #類別多
    data_less = df[df['income']==1]  #類別少
    np.random.seed(61)
    index = np.random.randint(len(data_more),size = percent*(len(df)-len(data_more)))
    lower_data_more = data_more.iloc[list(index)]   #下採樣
    undersampling_data = pd.concat([lower_data_more, data_less])
    return (undersampling_data)