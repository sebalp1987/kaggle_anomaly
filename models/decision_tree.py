import pandas as pd
import numpy as np

from resources import STRING, df_utils

from config.config import gral_parameters, oversampling

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, fbeta_score
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import ADASYN

df = pd.read_csv(STRING.output_train, sep=',', encoding='utf-8')


y = df.sort_values('TransactionDT')['isFraud']
df = df.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

df = df_utils.clean_inf_nan(df)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(df)
df = pd.DataFrame(imputer.transform(df), columns=df.columns.values.tolist())

# Model Parameters
len_df = len(df.index)
nfeatures = len(df.columns)

decision_tree = {
    'boosting_type':['gbdt', 'dart', 'goss'],
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': range(1, 10, 1),
    'min_samples_leaf': range(round(len_df*0.005), round(len_df*0.1), 100),
    'min_samples_split': range(round(len_df*0.01), round(len_df*0.2), 100),
    'max_features': [round(nfeatures/3), round(np.sqrt(nfeatures)), round(np.log(nfeatures)) + 1],
    'random_state': [gral_parameters.get('random_state')],
    'class_weight': [None, 'balanced'],
    'sampling': [None, 'Adasyn'],
    'n_jobs': -1

}

tresholds = np.linspace(0.01, 1.0, 1000)

# 1) Stratifield 5 CV Training Data
scores = []
y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=gral_parameters.get('random_state'))
model = DecisionTreeClassifier()

# grid search
for g in ParameterGrid(decision_tree):
    params = g.copy()
    del g['sampling']
    # stratified CV
    for train_index, test_index in skf.split(df.values, y.values):
        X_train, X_test = df.loc[train_index].values, df.loc[test_index].values
        y_train, y_test = y.loc[train_index].values, y.loc[test_index].values

        # oversampling
        if params.get('sampling') == 'Adasyn':
            g.update({'class_weight': None})
            ovs_model = ADASYN().set_params(**oversampling)
            X_train, y_train = ovs_model.fit_sample(X_train, y_train)

        model.set_params(**g)
        model.fit(X_train, y_train)

        y_pred_score_i = model.predict_proba(X_test)
        y_pred_score = np.append(y_pred_score, y_pred_score_i, axis=0)
        predicted_index = np.append(predicted_index, test_index, axis=0)
        del X_train, X_test, y_train, y_test

    # model prediction
    y_pred_score = np.delete(y_pred_score, 0, axis=1)

    # performance evaluation
    for treshold in tresholds:
        y_hat = (y_pred_score > treshold).astype(int)
        y_hat = y_hat.tolist()
        y_hat = [item for sublist in y_hat for item in sublist]

        scores.append([
            recall_score(y_pred=y_hat, y_true=y.values),
            precision_score(y_pred=y_hat, y_true=y.values),
            fbeta_score(y_pred=y_hat, y_true=y.values,
                        beta=gral_parameters.get('fbeta_score'))])

    scores = np.array(scores)
    precision = scores[:, 1].argmax()
    recall = scores[:, 0].argmax()
    fbeta = scores[:, 2].argmax()
    final_tresh = tresholds[scores[:, 2].argmax()]

    # save results
    try:
        df_monitor = pd.read_csv(STRING.monitoring_performance, sep=',', encoding='utf-8')
    except FileNotFoundError:
        df_monitor = pd.DataFrame(columns=['model', 'parameters', 'treshold', 'precision', 'recall', 'fbscore'])

    df_monitor = df_monitor.append(pd.DataFrame([['BT', params, final_tresh, precision, recall, fbeta]],
                                                columns=['model', 'parameters', 'treshold', 'precision', 'recall',
                                                         'fbscore']))
    df_monitor.to_csv(STRING.monitoring_performance, sep=',', index=False)

