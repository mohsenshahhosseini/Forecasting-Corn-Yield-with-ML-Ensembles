# -*- coding: utf-8 -*-
"""

@author: Mohsen
Ensemble Optimization - Bayesian Search - USDA Corn Yield Prediction
"""
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import linear_model
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
import warnings
from scipy.io import loadmat
from sklearn.model_selection import LeavePGroupsOut, GridSearchCV, GroupKFold
from sklearn.externals import joblib
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import os



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
np.random.seed(1369)

population = loadmat('INFO_POPULATION.mat')
progress = loadmat('INFO_PROGRESS.mat')
soil = loadmat('INFO_SOIL.mat')
weather = loadmat('INFO_Weather2.mat')
Yield = loadmat('INFO_Yield.mat')

population = population['INFO_POPULATION']
progress = progress['INFO_PROGRESS']
soil = soil['INFO_SOIL']
weather = weather['INFO_Weather']
weather = weather[:,119:308]
Yield = Yield['INFO_Yield']

data_d = pd.DataFrame({'Yield': Yield[:,3],
                       'Year': Yield[:,0].astype('int'),
                       'State': Yield[:,1].astype('int'),
                       'County': Yield[:,2].astype('int'),
                       'Population': population[:,3].astype('int')})
data_d = data_d[data_d.Year>=2000]
data_d = data_d.reset_index(drop=True)

data_d['yield_trend'] = 0
for s in data_d.State.unique():
    for c in data_d[data_d.State==s].County.unique():
        y1 = pd.DataFrame(data_d.Yield[(data_d.Year<2018) & ((data_d.State).astype('int') == s) & ((data_d.County).astype('int') == c)])
        x1 = pd.DataFrame(data_d.Year[(data_d.Year<2018) & ((data_d.State).astype('int') == s) & ((data_d.County).astype('int') == c)])
        regressor = LinearRegression()
        regressor.fit(x1, y1)
        data_d['yield_trend'][(data_d.Year<2018)&(data_d.State==s)&(data_d.County==c)] = regressor.predict(x1)
        if len(data_d.Year[(data_d.Year==2018)&(data_d.State==s)&(data_d.County==c)].unique()) != 0:
            data_d['yield_trend'][(data_d.Year==2018)&(data_d.State==s)&(data_d.County==c)] = regressor.predict(pd.DataFrame([2018]))

data_d['yield_avg'] = 0
for y in data_d.Year[data_d.Year<2018].unique():
    for s in data_d.State.unique():
        data_d.yield_avg[(data_d.Year == y) & (data_d.State == s)] = data_d.Yield[(data_d.Year == y) & (data_d.State == s)].mean()
        data_d.yield_avg[(data_d.Year == 2018) & (data_d.State == s)] = data_d.Yield[(data_d.Year == 2017) & (data_d.State == s)].mean()

data_d = data_d.sort_values(by='Year')

data = pd.concat([data_d,pd.DataFrame(progress[:,12:25])], axis=1)
data = pd.concat([data,pd.DataFrame(weather)],axis=1)
data = pd.concat([data,pd.DataFrame(soil)],axis=1)

data = data.dropna()
data = data.reset_index(drop=True)

progress_names = ['Progress_' + str(i) for i in range(1,14)]                                        # renaming columns
weather_names = ['Weather_' + str(i) for i in range(1,190)]
soil_names = ['Soil_' + str(i) for i in range(1,181)]
names = ['Yield' , 'Year', 'State', 'County', 'Population', 'yield_trend', 'yield_avg', progress_names, weather_names, soil_names]
names[7:10] = [item for sublist in names[7:10] for item in sublist]
data.columns = names

columns_to_scale = data.drop(columns=['Yield','Year','State','County']).columns.values
scaler = MinMaxScaler()
scaled_columns = scaler.fit_transform(data[columns_to_scale])
scaled_columns = pd.DataFrame(scaled_columns, columns=columns_to_scale)

data2 = pd.DataFrame(data.Yield)
data = pd.concat([data2, data.Year, scaled_columns], axis=1)
data = data.sort_values(by='Year')

test = data[data.Year==2018]
train = data[data.Year<2018]

x_test = test.drop(columns=['Yield'])
y_test = test.Yield

X = train.drop(columns=['Yield'])
X = X.reset_index(drop=True)
Y = train.Yield
Y.reset_index(inplace=True, drop=True)


# feature selection with random forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, Y)

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf, cv=20, n_iter=100).fit(X, Y)
feature_importances = [(feature, importance) for feature, importance in zip(list(X.columns), list(np.abs(perm.feature_importances_)))]
feature_importances = pd.DataFrame(sorted(feature_importances, key = lambda x: x[1], reverse = True))
selected_features = feature_importances.iloc[0:80,:][0]
if np.isin('Year', selected_features)==False:
    selected_features = selected_features.append(pd.Series('Year'))
X = X.loc[:,selected_features]
x_test = x_test.loc[:,selected_features]


# feature selection with correlation
corr = X.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns].values
if np.isin('Year', selected_columns)==False:
    selected_columns = np.append(selected_columns,'Year')
X = X.loc[:,selected_columns]
x_test = x_test.loc[:,selected_columns]



 ## ---------------- Bayesian Search ---------------- ##


max_evals = 20

def objective_LASSO(params):
    LASSO_df_B = pd.DataFrame()
    L1_B = Lasso()
    for train_index, test_index in myCV:
        LASSO_B = L1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        LASSO_df_B = pd.concat([LASSO_df_B, pd.DataFrame(LASSO_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_LASSO = mse(data_d.Yield[(data_d.Year > 2007) & (data_d.Year <= 2017)], LASSO_df_B)
    return {'loss': loss_LASSO, 'params': params, 'status': STATUS_OK}

space_LASSO = {'alpha': hp.uniform('alpha', 10**-5, 1)}
tpe_algorithm = tpe.suggest
trials_LASSO = Trials()
best_LASSO = fmin(fn=objective_LASSO, space=space_LASSO, algo=tpe.suggest,
                  max_evals=max_evals, trials=trials_LASSO, rstate=np.random.RandomState(1369))
LASSO_param_B = pd.DataFrame({'alpha': []})
for i in range(max_evals):
    LASSO_param_B.alpha[i] = trials_LASSO.results[i]['params']['alpha']
LASSO_param_B = pd.DataFrame(LASSO_param_B.alpha)



def objective_XGB(params):
    XGB_df_B = pd.DataFrame()
    X1_B = XGBRegressor(objective='reg:linear', **params)
    for train_index, test_index in myCV:
        XGB_B = X1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        XGB_df_B = pd.concat([XGB_df_B, pd.DataFrame(X1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_XGB = mse(data_d.Yield[(data_d.Year > 2007) & (data_d.Year <= 2017)], XGB_df_B)
    return {'loss': loss_XGB, 'params': params, 'status': STATUS_OK}

space_XGB = {'gamma': hp.uniform('gamma', 0, 1),
             'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
             'n_estimators': hp.choice('n_estimators', [100, 300, 500, 1000]),
             'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(3, 20, 1)])}
tpe_algorithm = tpe.suggest
trials_XGB = Trials()
best_XGB = fmin(fn=objective_XGB, space=space_XGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_XGB, rstate=np.random.RandomState(1369))
XGB_param_B = pd.DataFrame({'gamma': [], 'learning_rate': [], 'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    XGB_param_B.gamma[i] = trials_XGB.results[i]['params']['gamma']
    XGB_param_B.learning_rate[i] = trials_XGB.results[i]['params']['learning_rate']
    XGB_param_B.n_estimators[i] = trials_XGB.results[i]['params']['n_estimators']
    XGB_param_B.max_depth[i] = trials_XGB.results[i]['params']['max_depth']
XGB_param_B = pd.DataFrame({'gamma': XGB_param_B.gamma,
                            'learning_rate': XGB_param_B.learning_rate,
                            'n_estimators': XGB_param_B.n_estimators,
                            'max_depth': XGB_param_B.max_depth})


def objective_LGB(params):
    LGB_df_B = pd.DataFrame()
    G1_B = LGBMRegressor(objective='regression', **params)
    for train_index, test_index in myCV:
        LGB_B = G1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        LGB_df_B = pd.concat([LGB_df_B, pd.DataFrame(G1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_LGB = mse(data_d.Yield[(data_d.Year > 2007) & (data_d.Year <= 2017)], LGB_df_B)
    return {'loss': loss_LGB, 'params': params, 'status': STATUS_OK}

space_LGB = {'num_leaves': hp.choice('num_leaves', [int(x) for x in np.arange(5, 40, 2)]),
             'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
             'n_estimators': hp.choice('n_estimators', [500, 1000, 1500, 2000])}
tpe_algorithm = tpe.suggest
trials_LGB = Trials()
best_LGB = fmin(fn=objective_LGB, space=space_LGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_LGB, rstate=np.random.RandomState(1369))
LGB_param_B = pd.DataFrame({'num_leaves': [], 'learning_rate': [], 'n_estimators': []})
for i in range(max_evals):
    LGB_param_B.num_leaves[i] = trials_LGB.results[i]['params']['num_leaves']
    LGB_param_B.learning_rate[i] = trials_LGB.results[i]['params']['learning_rate']
    LGB_param_B.n_estimators[i] = trials_LGB.results[i]['params']['n_estimators']
LGB_param_B = pd.DataFrame({'num_leaves': LGB_param_B.num_leaves,
                            'learning_rate': LGB_param_B.learning_rate,
                            'n_estimators': LGB_param_B.n_estimators})


def objective_RF(params):
    RF_df_B = pd.DataFrame()
    R1_B = RandomForestRegressor(**params)
    for train_index, test_index in myCV:
        RF_B = R1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        RF_df_B = pd.concat([RF_df_B, pd.DataFrame(R1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_RF = mse(data_d.Yield[(data_d.Year > 2007) & (data_d.Year <= 2017)], RF_df_B)
    return {'loss': loss_RF, 'params': params, 'status': STATUS_OK}

space_RF = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
            'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(5, 41, 5)])}
tpe_algorithm = tpe.suggest
trials_RF = Trials()
best_RF = fmin(fn=objective_RF, space=space_RF, algo=tpe.suggest,
               max_evals=max_evals, trials=trials_RF, rstate=np.random.RandomState(1369))
RF_param_B = pd.DataFrame({'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    RF_param_B.n_estimators[i] = trials_RF.results[i]['params']['n_estimators']
    RF_param_B.max_depth[i] = trials_RF.results[i]['params']['max_depth']
RF_param_B = pd.DataFrame({'n_estimators': RF_param_B.n_estimators,
                           'max_depth': RF_param_B.max_depth})




## ---------------- Building models ---------------- ##


LASSO_df2 = pd.DataFrame()
L2 = Lasso(alpha=trials_LASSO.best_trial['result']['params']['alpha'], random_state=1369)
start_ens1 = time.time()
for train_index, test_index in myCV:
    L2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LASSO_df2 = pd.concat([LASSO_df2, pd.DataFrame(L2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LASSO_df2 = LASSO_df2.reset_index(drop=True)
time_ens1 = time.time() - start_ens1
LASSO_mse2 = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], LASSO_df2)
start_train_LASSO = time.time()
LASSO = L2.fit(X.drop(columns='Year'), Y)
train_time_LASSO = time.time() - start_train_LASSO
start_predict_LASSO = time.time()
LASSO_preds_test2 = LASSO.predict(x_test.drop(columns='Year'))
predict_time_LASSO = time.time() - start_predict_LASSO
pd.DataFrame(LASSO_preds_test2).to_csv('LASSO_preds_test.csv')
LASSO_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LASSO_preds_test2)
LASSO_rmse_test2 = np.sqrt(LASSO_mse_test2)


### ---------- XGB ------------ ###
XGB_df2 = pd.DataFrame()
X2 = XGBRegressor(objective='reg:linear',
                  gamma=trials_XGB.best_trial['result']['params']['gamma'],
                  learning_rate=trials_XGB.best_trial['result']['params']['learning_rate'],
                  n_estimators=int(trials_XGB.best_trial['result']['params']['n_estimators']),
                  max_depth=int(trials_XGB.best_trial['result']['params']['max_depth']), random_state=1369)
start_ens2 = time.time()
for train_index, test_index in myCV:
    X2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    XGB_df2 = pd.concat([XGB_df2, pd.DataFrame(X2.predict(np.array(X.drop(columns='Year'))[test_index]))])
XGB_df2 = XGB_df2.reset_index(drop=True)
time_ens2 = time.time() - start_ens2
XGB_mse2 = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], XGB_df2)
start_train_XGB = time.time()
XGB = X2.fit(X.drop(columns='Year'), Y)
train_time_XGB = time.time() - start_train_XGB
start_predict_XGB = time.time()
XGB_preds_test2 = XGB.predict(x_test.drop(columns='Year'))
predict_time_XGB = time.time() - start_predict_XGB
pd.DataFrame(XGB_preds_test2).to_csv('XGB_preds_test.csv')
XGB_mse_test2 = mse(data_d.Yield[data_d.Year==2018], XGB_preds_test2)
XGB_rmse_test2 = np.sqrt(XGB_mse_test2)


### ---------- LGB ------------ ###
LGB_df2 = pd.DataFrame()
G2 = LGBMRegressor(objective='regression', random_state=1369,
                   num_leaves=int(trials_LGB.best_trial['result']['params']['num_leaves']),
                   learning_rate=trials_LGB.best_trial['result']['params']['learning_rate'],
                   n_estimators=int(trials_LGB.best_trial['result']['params']['n_estimators']))
start_ens3 = time.time()
for train_index, test_index in myCV:
    G2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LGB_df2 = pd.concat([LGB_df2, pd.DataFrame(G2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LGB_df2 = LGB_df2.reset_index(drop=True)
time_ens3 = time.time() - start_ens3
LGB_mse2 = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], LGB_df2)
start_train_LGB = time.time()
LGB = G2.fit(X.drop(columns='Year'), Y)
train_time_LGB = time.time() - start_train_LGB
start_predict_LGB = time.time()
LGB_preds_test2 = LGB.predict(x_test.drop(columns='Year'))
predict_time_LGB = time.time() - start_predict_LGB
pd.DataFrame(LGB_preds_test2).to_csv('LGB_preds_test.csv')
LGB_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LGB_preds_test2)
LGB_rmse_test2 = np.sqrt(LGB_mse_test2)


### ---------- RF ------------ ###
RF_df2 = pd.DataFrame()
R2 = RandomForestRegressor(max_depth=int(trials_RF.best_trial['result']['params']['max_depth']),
                           n_estimators=int(trials_RF.best_trial['result']['params']['n_estimators']), random_state=1369)
start_ens4 = time.time()
for train_index, test_index in myCV:
    R2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    RF_df2 = pd.concat([RF_df2, pd.DataFrame(R2.predict(np.array(X.drop(columns='Year'))[test_index]))])
RF_df2 = RF_df2.reset_index(drop=True)
time_ens4 = time.time() - start_ens4
RF_mse2 = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], RF_df2)
start_train_RF = time.time()
RF = R2.fit(X.drop(columns='Year'), Y)
train_time_RF = time.time() - start_train_RF
start_predict_RF = time.time()
RF_preds_test2 = RF.predict(x_test.drop(columns='Year'))
predict_time_RF = time.time() - start_predict_RF
pd.DataFrame(RF_preds_test2).to_csv('RF_preds_test.csv')
RF_mse_test2 = mse(data_d.Yield[data_d.Year==2018], RF_preds_test2)
RF_rmse_test2 = np.sqrt(RF_mse_test2)


### ---------- LR ------------ ###
LR_df2 = pd.DataFrame()
lm2 = LinearRegression()
lm2.fit(X.drop(columns='Year'),Y)
start_ens5 = time.time()
for train_index, test_index in myCV:
    lm2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LR_df2 = pd.concat([LR_df2, pd.DataFrame(lm2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LR_df2 = LR_df2.reset_index(drop=True)
time_ens5 = time.time() - start_ens5
LR_mse2 = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], LR_df2)
start_train_LR = time.time()
LR = lm2.fit(X.drop(columns='Year'), Y)
train_time_LR = time.time() - start_train_LR
start_predict_LR = time.time()
LR_preds_test2 = LR.predict(x_test.drop(columns='Year'))
predict_time_LR = time.time() - start_predict_LR
pd.DataFrame(LR_preds_test2).to_csv('LR_preds_test.csv')
LR_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LR_preds_test2)
LR_rmse_test2 = np.sqrt(LR_mse_test2)



## ---------------- Optimizing Ensembles ---------------- ##

start_train_cowe = time.time()
def objective2(y):
    return mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)],
               (y[0]*LASSO_df2 + y[1]*XGB_df2 + y[2]*LGB_df2 + y[3]*RF_df2 + y[4]*LR_df2))

def constraint12(y):
    return y[0] + y[1] + y[2] + y[3] + y[4] - 1.0
def constraint22(y):
    return LASSO_mse2 - objective2(y)
def constraint32(y):
    return XGB_mse2 - objective2(y)
def constraint42(y):
    return LGB_mse2 - objective2(y)
def constraint52(y):
    return RF_mse2 - objective2(y)
def constraint62(y):
    return LR_mse2 - objective2(y)


y0 = np.zeros(5)
y0[0] = 1 / 5
y0[1] = 1 / 5
y0[2] = 1 / 5
y0[3] = 1 / 5
y0[4] = 1 / 5

b = (0, 1.0)
bnds2 = (b, b, b, b, b)
con12 = {'type': 'eq', 'fun': constraint12}
con22 = {'type': 'ineq', 'fun': constraint22}
con32 = {'type': 'ineq', 'fun': constraint32}
con42 = {'type': 'ineq', 'fun': constraint42}
con52 = {'type': 'ineq', 'fun': constraint52}
con62 = {'type': 'ineq', 'fun': constraint62}

cons2 = [con12, con22, con32, con42, con52, con62]

solution2 = minimize(objective2, y0, method='SLSQP',
                    options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds2,
                    constraints=cons2)
y = solution2.x

train_time_cowe = time.time() - start_train_cowe + time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5
train_time_cls = time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5

start_predict_cowe = time.time()
cowe_preds_test = y[0]*LASSO_preds_test2 + y[1]*XGB_preds_test2 + y[2]*LGB_preds_test2 + y[3]*RF_preds_test2 + y[4]*LR_preds_test2
predict_time_cowe = time.time() - start_predict_cowe
cowe_mse_test = mse(data_d.Yield[data_d.Year==2018], cowe_preds_test)
cowe_rmse_test = np.sqrt(cowe_mse_test)
pd.DataFrame(cowe_preds_test).to_csv('cowe_preds_test.csv')


cowe_preds_CV = y[0]*LASSO_df2 + y[1]*XGB_df2 + y[2]*LGB_df2 + y[3]*RF_df2 + y[4]*LR_df2
cowe_mse_CV = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], cowe_preds_CV)
cowe_rmse_CV = np.sqrt(cowe_mse_CV)


start_predict_cls = time.time()
cls_preds_test = y0[0]*LASSO_preds_test2 + y0[1]*XGB_preds_test2 + y0[2]*LGB_preds_test2 + y0[3]*RF_preds_test2 + y0[4]*LR_preds_test2
predict_time_cls = time.time() - start_predict_cls
cls_mse_test = mse(data_d.Yield[data_d.Year==2018], cls_preds_test)
cls_rmse_test = np.sqrt(cls_mse_test)
pd.DataFrame(cls_preds_test).to_csv('cls_preds_test3.csv')


cls_preds_CV = y0[0]*LASSO_df2 + y0[1]*XGB_df2 + y0[2]*LGB_df2 + y0[3]*RF_df2 + y0[4]*LR_df2
cls_mse_CV = mse(data_d.Yield[(data_d.Year>2007) & (data_d.Year<=2017)], cls_preds_CV)
cls_rmse_CV = np.sqrt(cls_mse_CV)



## -------------------------------- STACKING -------------------------------- ##

predsDF2 = pd.DataFrame()
predsDF2['LASSO'] = LASSO_df2[0]
predsDF2['XGB']= XGB_df2[0]
predsDF2['LGB'] = LGB_df2[0]
predsDF2['RF'] = RF_df2[0]
predsDF2['LR'] = LR_df2[0]
predsDF2['Y'] = data_d.Yield[(data_d.Year > 2007) & (data_d.Year <= 2017)].reset_index(drop=True)
x_stacked2 = predsDF2.drop(columns='Y', axis=1)
y_stacked2 = predsDF2['Y']
testPreds2 = pd.DataFrame([LASSO_preds_test2, XGB_preds_test2, LGB_preds_test2, RF_preds_test2, LR_preds_test2]).T
testPreds2.columns = ['LASSO', 'XGB', 'LGB', 'RF', 'LR']


stck_reg2 = LinearRegression()
start_train_stck_reg = time.time()
stck_reg2.fit(x_stacked2, y_stacked2)
train_time_stck_reg = time.time() - start_train_stck_reg + time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5
start_predict_stck_reg = time.time()
stck_reg_preds_test2 = stck_reg2.predict(testPreds2)
predict_time_stck_reg = time.time() - start_predict_stck_reg
stck_reg_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_reg_preds_test2)
stck_reg_rmse_test2 = np.sqrt(stck_reg_mse_test2)
pd.DataFrame(stck_reg_preds_test2).to_csv('stck_reg_preds_test.csv')

stck_lasso2 = Lasso()
start_train_stck_lasso = time.time()
stck_lasso2.fit(x_stacked2, y_stacked2)
train_time_stck_lasso = time.time() - start_train_stck_lasso + time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5
start_predict_stck_lasso = time.time()
stck_lasso_preds_test2 = stck_lasso2.predict(testPreds2)
predict_time_stck_lasso = time.time() - start_predict_stck_lasso
stck_lasso_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_lasso_preds_test2)
stck_lasso_rmse_test2 = np.sqrt(stck_lasso_mse_test2)
pd.DataFrame(stck_lasso_preds_test2).to_csv('stck_lasso_rmse_test.csv')

stck_rf2 = RandomForestRegressor()
start_train_stck_rf = time.time()
stck_rf2.fit(x_stacked2, y_stacked2)
train_time_stck_rf = time.time() - start_train_stck_rf + time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5
start_predict_stck_rf = time.time()
stck_rf_preds_test2 = stck_rf2.predict(testPreds2)
predict_time_stck_rf = time.time() - start_predict_stck_rf
stck_rf_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_rf_preds_test2)
stck_rf_rmse_test2 = np.sqrt(stck_rf_mse_test2)
pd.DataFrame(stck_rf_preds_test2).to_csv('stck_rf_rmse_test.csv')

stck_lgb2 = LGBMRegressor()
start_train_stck_lgb = time.time()
stck_lgb2.fit(x_stacked2, y_stacked2)
train_time_stck_lgb = time.time() - start_train_stck_lgb + time_ens1 + time_ens2 + time_ens3 + time_ens4 + time_ens5
start_predict_stck_lgb = time.time()
stck_lgb_preds_test2 = stck_lgb2.predict(testPreds2)
predict_time_stck_lgb = time.time() - start_predict_stck_lgb
stck_lgb_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_lgb_preds_test2)
stck_lgb_rmse_test2 = np.sqrt(stck_lgb_mse_test2)
pd.DataFrame(stck_lgb_preds_test2).to_csv('stck_lgb_rmse_test.csv')




## -------------------------- RESULTS -------------------------- ##


test_results = pd.DataFrame(data={'model':['RMSE'],'LASSO':[LASSO_rmse_test2], 'XGB':[XGB_rmse_test2], 'LGB':[LGB_rmse_test2],
                                  'RF': [RF_rmse_test2], 'LR': [LR_rmse_test2],
                                  'COWE': [cowe_rmse_test], 'Classical': [cls_rmse_test],
                                  'stck_reg': [stck_reg_rmse_test2], 'stck_lasso': [stck_lasso_rmse_test2],
                                  'stck_rf': [stck_rf_rmse_test2], 'stck_lgb': [stck_lgb_rmse_test2]})

CV_results = pd.DataFrame(data={'model':['RMSE'], 'LASSO':[np.sqrt(LASSO_mse2)], 'XGB':[np.sqrt(XGB_mse2)],
                                'LGB':[np.sqrt(LGB_mse2)], 'RF': [np.sqrt(RF_mse2)], 'LR': [np.sqrt(LR_mse2)],
                                'COWE': [cowe_rmse_CV],
                                'Classical':[cls_rmse_CV]})

train_times = pd.DataFrame(data={'model':['Training time'], 'LASSO':[train_time_LASSO], 'XGB':[train_time_XGB],
                                 'LGB':[train_time_LGB], 'RF': [train_time_RF], 'LR': [train_time_LR],
                                  'COWE': [train_time_cowe], 'Classical': [train_time_cls],
                                  'stck_reg': [train_time_stck_reg], 'stck_lasso': [train_time_stck_lasso],
                                  'stck_rf': [train_time_stck_rf], 'stck_lgb': [train_time_stck_lgb]})

predict_times = pd.DataFrame(data={'model':['Prediction time'], 'LASSO':[predict_time_LASSO], 'XGB':[predict_time_XGB],
                                 'LGB':[predict_time_LGB], 'RF': [predict_time_RF], 'LR': [predict_time_LR],
                                  'COWE': [predict_time_cowe], 'Classical': [predict_time_cls],
                                  'stck_reg': [predict_time_stck_reg], 'stck_lasso': [predict_time_stck_lasso],
                                  'stck_rf': [predict_time_stck_rf], 'stck_lgb': [predict_time_stck_lgb]})

test_results.to_csv('test.csv')
CV_results.to_csv('CV.csv')
train_times.to_csv('TIME_train_times.csv')
predict_times.to_csv('TIME_predict_times.csv')



## -------------------------- PDP -------------------------- ##

from pdpbox import pdp

selected_columns = [x for x in selected_columns if x != 'Year']
pdp_ensemble = pd.DataFrame({'x':[], 'pdp':[]})

for i in range(len(selected_columns)):
    pdp_lasso = pdp.pdp_isolate(model=LASSO,
                                dataset=train.loc[:,selected_columns],
                                model_features=selected_columns,
                                feature=selected_columns[i])

    pdp_xgb = pdp.pdp_isolate(model=XGB,
                              dataset=train.loc[:,selected_columns],
                              model_features=selected_columns,
                              feature=selected_columns[i])

    pdp_lgb = pdp.pdp_isolate(model=LGB,
                              dataset=train.loc[:,selected_columns],
                              model_features=selected_columns,
                              feature=selected_columns[i])

    pdp_rf = pdp.pdp_isolate(model=RF,
                             dataset=train.loc[:,selected_columns],
                             model_features=selected_columns,
                             feature=selected_columns[i])

    pdp_lr = pdp.pdp_isolate(model=LR,
                             dataset=train.loc[:,selected_columns],
                             model_features=selected_columns,
                             feature=selected_columns[i])

    pdp_ensemble2 = pd.DataFrame({'x':pdp_lasso.display_columns,
                                  'pdp':y[0]*pdp_lasso.pdp + y[1]*pdp_xgb.pdp + y[2]*pdp_lgb.pdp + y[3]*pdp_rf.pdp + y[4]*pdp_lr.pdp})

    pdp_ensemble = pd.concat([pdp_ensemble, pdp_ensemble2], axis=0)

pd.DataFrame(pdp_ensemble).to_csv('pdp_ensemble.csv')
pd.DataFrame(selected_columns).to_csv('selected_features.csv')


