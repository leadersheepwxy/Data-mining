import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 加載你的數據
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# 創建成lgb特徵的數據集格式
lgb_train = lgb.Dataset(X_train, y_train)  # 將數據保存到LightGBM二進制文件將使加載更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 創建驗證數據

# 將參數寫成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',# 設置提升類型
    'objective': 'regression',  # 目標函數
    'metric': {'rmse', 'auc'},# 葉子節點數
    'metric_freq': 1,
    'num_trees': 100,
    'num_leaves': 16,
    'max_depth': 4, #設置數據的深度，防止過擬合
    'learning_rate': 0.02, # 學習速率
    'feature_fraction': 0.5,# 建樹的特徵選擇比例
    'feature_fraction_seed': 6,
    'bagging_fraction': 0.5,# 建樹的樣本採樣比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代執行bagging
    'is_unbalance': 'true', #數據集如果樣本不均衡，可以明顯提高準確率
    'verbose': 0
}
evals_result = {}
print('Start training...')
# 訓練 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5,evals_result=evals_result)  # 訓練數據需要參數列表和數據集
print('Save model...')
gbm.save_model('model.txt')  # 訓練後保存模型到文件
print('Start predicting...')
# 預測數據集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在訓練期間啓用了早期停止，可以通過best_iteration方式從最佳迭代中獲得預測
# 評估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 計算真實值和預測值之間的均方根誤差