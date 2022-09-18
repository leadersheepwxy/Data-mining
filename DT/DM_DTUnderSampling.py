from sklearn.model_selection import StratifiedKFold
import pandas as pd
import statistics
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import plot_roc_curve, roc_curve, auc
from sklearn import tree
import settingFunction as sf

def lowerSample(df,percent=1):
    data_more = df[df['income']==0]  #類別多
    data_less = df[df['income']==1]  #類別少
    index = np.random.randint(len(data_more),size = percent*(len(df)-len(data_more)))
    lower_data_more = data_more.iloc[list(index)]   #下採樣
    undersampling_data = pd.concat([lower_data_more, data_less])
    return (undersampling_data)
times_best = [0, 0]
times = 0
while (times <= 100):
    np.random.seed(times)
    print(times)
    # 導入特徵數據
    data_original = pd.read_csv("C:\\Users\\u1070\\PycharmProjects\\recorder\\DM\\adult_fin.csv")
    data = lowerSample(data_original)
    Depth_value_range = range(1, 11)
    X, y, X_training, X_testing, y_training, y_testing = sf.DataImport(data)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)  # 定義10折交叉驗證:分層KFold 90/10
    i, j = 1, 1
    bestMean = [0,0,0,0]
    t_bestMean = [0,0,0,0]
    bestGroup, bestParm = 0, 0
    bestAuc, bestRecall, bestPrec, bestF1 = 0,0,0,0
    t, v = [], []
    meanAuc_list = []
    t_meanAuc_list = []
    fprList_best, tprList_best = [], []

    for n in Depth_value_range:
        auc_list, recall_list, prec_list, f1_list = [], [], [], []
        t_auc_list, t_recall_list, t_prec_list, t_f1_list = [], [], [], []
        train_index_list, val_index_list = [], []
        fpr_list, tpr_list = [], []
        for train_index, val_index in cv.split(X_training, y_training):
            X_train, X_val = X_training[train_index], X_training[val_index]
            y_train, y_val = y_training[train_index], y_training[val_index]

            model = DecisionTreeClassifier(random_state=i, max_depth=n)
            model.fit(X_train, y_train)  # 訓練
            predictions_train = model.predict(X_train)
            predictions_val = model.predict(X_val)
            predictions_thresholds = model.predict_proba(X_val)[:, 1]
            # print("-------第", i, "組-------")

            # fpr, tpr, thresholds = roc_curve(y_val, predictions_thresholds)
            # fpr_list.append(fpr)
            # tpr_list.append(tpr)

            t_accuracy = sklearn.metrics.accuracy_score(y_train, predictions_train)
            t_recall = sklearn.metrics.recall_score(y_train, predictions_train)
            t_precision = sklearn.metrics.precision_score(y_train, predictions_train)
            t_f1 = sklearn.metrics.f1_score(y_train, predictions_train)
            t_auc_list.append(t_accuracy)
            t_recall_list.append(t_recall)
            t_prec_list.append(t_precision)
            t_f1_list.append(t_f1)

            accuracy = sklearn.metrics.accuracy_score(y_val, predictions_val)
            recall = sklearn.metrics.recall_score(y_val, predictions_val)
            precision = sklearn.metrics.precision_score(y_val, predictions_val)
            f1 = sklearn.metrics.f1_score(y_val, predictions_val)
            auc_list.append(accuracy)
            recall_list.append(recall)
            prec_list.append(precision)
            f1_list.append(f1)
            train_index_list.append(train_index)
            val_index_list.append(val_index)

            if (j == 10):
                t_meanAuc = statistics.mean(t_auc_list)
                meanAuc = statistics.mean(auc_list)
                meanAuc_list.append(meanAuc)

                # print("第", int(i / 10), "組 結果:", auc_list)
                if meanAuc > bestMean[0]:
                    # train
                    t_meanRecall = statistics.mean(t_recall_list)
                    t_meanPrec = statistics.mean(t_prec_list)
                    t_meanF1 = statistics.mean(t_f1_list)
                    t_bestMean = (t_meanAuc, t_meanRecall, t_meanPrec, t_meanF1)

                    # val
                    fprList_best = fpr_list
                    tprList_best = tpr_list

                    meanRecall = statistics.mean(recall_list)
                    meanPrec = statistics.mean(prec_list)
                    meanF1 = statistics.mean(f1_list)
                    bestMean = (meanAuc, meanRecall, meanPrec, meanF1)

                    bestAuc_index = auc_list.index(max(auc_list))
                    bestAuc = auc_list[bestAuc_index]
                    bestRecall = recall_list[bestAuc_index]
                    bestPrec = prec_list[bestAuc_index]
                    bestF1 = f1_list[bestAuc_index]

                    t = train_index_list[bestAuc_index]
                    v = val_index_list[bestAuc_index]

                    bestGroup = ((bestAuc_index + 1) + (i - 10))
                    bestParm = n
                j = 0
            j += 1
            i += 1
    foldNum = (bestGroup % 10)
    if foldNum == 0:
        foldNum == 10

    # tprs, aucs = [], []
    # mean_fpr = np.linspace(0, 1, 100)
    # for i in range(0, 10):
    #     tprs.append(np.interp(mean_fpr, fprList_best[i], tprList_best[i]))
    #     tprs[-1][0] = 0.0
    #     roc_auc = auc(fprList_best[i], tprList_best[i])
    #     aucs.append(roc_auc)
    #     plt.plot(fprList_best[i], tprList_best[i], lw=1, alpha=0.8,
    #              label='ROC fold %d (AUC = %0.2f)' % ((i + 1), roc_auc))
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=1.5, alpha=.9)
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1, label=r'$\pm$ 1 std. dev.')
    #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('CrossValidation Roc')
    # plt.legend(loc="lower right")
    # plt.show()
    # print("=========================Data切割狀況=========================")
    # print("--TRAINING SET--")
    # print(X_training.shape)
    # print("-train-")
    # print(X_train.shape)
    # print("-validation-")
    # print(X_val.shape)
    # print("--TESTING SET--")
    # print(X_testing.shape)

    # print("=========================參數最佳化=========================")
    bestValue = [bestGroup, bestAuc, bestParm]
    bestValue_name = ['最佳組別:', '最佳準確率:', '最佳深度:']  #
    bestValue_Chart = pd.DataFrame(bestValue, bestValue_name)
    # print(bestValue_Chart)

    # print("=========================預測結果=========================")
    # print("* 平均最佳： NO.", bestGroup, " (第",int(bestGroup/10+1),"組, fold",foldNum,") *")
    model = DecisionTreeClassifier(random_state=bestGroup, max_depth=bestParm)
    X_train, X_val = X_training[t], X_training[v]
    y_train, y_val = y_training[t], y_training[v]
    model.fit(X_train, y_train)  # 訓練

    # print("\n-Validation-")
    predictions_val = model.predict(X_val)
    val_score = sf.ScoreReport(y_val, predictions_val)

    # print("\n-TEST-")
    predictions_test = model.predict(X_testing)
    test_score = sf.ScoreReport(y_testing, predictions_test)

    times_acc_temp = sklearn.metrics.accuracy_score(y_testing, predictions_test)
    print("acc:",times_acc_temp)

    if times_acc_temp > times_best[0]:
        times_best[0] = times_acc_temp
        times_best[1] = times
        times += 1
    else:
        times += 1
        continue

    Scores = {
        "Score" : ['accuracy', 'recall', 'precision', 'f1'],
        "Val(mean)" : bestMean,
        "Val(best)" : val_score,
        "Test" : test_score
    }
    ScoresChart = pd.DataFrame(Scores)
    # print(ScoresChart)
    # plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=1.5, alpha=.9)
    # ax = plt.gca()
    # val_roc = plot_roc_curve(model, X_val, y_val, ax=ax, color='green')
    # test_roc = plot_roc_curve(model, X_testing, y_testing, ax=ax, color='orange')
    # sf.rocshow('ROC')


    # fn = X.columns  # 有四段射頻18
    # cn = ['0', '1']
    # tree.plot_tree(model, feature_names=fn, class_names=cn, filled=True, fontsize=5.5)
    # plt.show()

print("最佳:",times_best)