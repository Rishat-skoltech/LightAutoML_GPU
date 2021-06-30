from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from numpy_impl import numpy_impl as nn
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from time import perf_counter
import os
from rapids_impl import rapids_impl as rr

##################################################################################################
##################################################################################################
t0 = perf_counter()
if (not os.path.exists('./data/HIGGS.parquet')):
    if (not os.path.exists('./data/HIGGS.csv')):
        print("you need to download HIGGS data set, call it HIGGS.csv (it's gonna be 8GB) and place it in the 'data' folder")
        exit()
    else:
        rr.to_parquet('./data', 'HIGGS')
data = pd.read_parquet("./data/HIGGS.parquet", engine='fastparquet')

t1 = perf_counter()
data_filled = nn.FillnaMedian_numpy(data.values.T)
t2 = perf_counter()
data = pd.DataFrame(data_filled.T, index=data.index, columns=data.columns)
y = data["label"]
X = data[data.columns.difference(["label"])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0x0BADBEEF)
t3 = perf_counter()
##################################################################################################
##################################################################################################
clf = CatBoostClassifier(iterations=100, task_type='GPU', devices='0')
#clf = CatBoostClassifier(iterations=100, task_type='GPU', devices='0:1:2:3')
clf.fit(X_train, y_train, verbose=False)
t4 = perf_counter()
probas = clf.predict_proba(X_test)
print("Logloss score for", clf.__class__.__name__, ":", log_loss(y_test, probas[:,-1]))
##################################################################################################
##################################################################################################
t5 = perf_counter()
sorted_ids = np.argsort(clf.feature_importances_)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.iloc[:, sorted_ids[:10]]), index=X_train.index, columns=X_train.columns[sorted_ids[:10]])
X_test = pd.DataFrame(scaler.fit_transform(X_test.iloc[:, sorted_ids[:10]]), index=X_test.index, columns=X_test.columns[sorted_ids[:10]])
t6 = perf_counter()
Lasso_model = Lasso()
Lasso_model.fit(X_train, y_train)
t7 = perf_counter()
res = Lasso_model.predict(X_test)
t8 = perf_counter()
print("Average precision score for", Lasso_model.__class__.__name__, ":", average_precision_score(y_test, res))

print("parquet", t1-t0)
print("fillnamedian", t2-t1)
print("prepare data", t3-t2)
print("catboost train time", t4-t3)
print("catboost inference time", t5-t4)
print("featureselect and scale", t6-t5)
print("lasso fit time", t7-t6)
print("lasso inference time", t8-t7)

