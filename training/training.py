import pandas as pd
import numpy as np
import os

from resources import STRING, df_utils

from config.config import gral_parameters, oversampling, lgb_optimal

import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import ADASYN

df = pd.read_csv(STRING.output_train, sep=',', encoding='utf-8')


y = df.sort_values('TransactionDT')['isFraud']
df = df.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

df = df_utils.clean_inf_nan(df)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(df)
df = pd.DataFrame(imputer.transform(df), columns=df.columns.values.tolist())

# 1) Stratifield 5 CV Training Data
scores = []
y_pred_score = np.empty(shape=[0, 2])
predicted_index = np.empty(shape=[0, ])
model = lgb.LGBMClassifier()
model.set_params(**lgb_optimal)


if gral_parameters.get('sampling') == 'Adasyn':
    ovs_model = ADASYN().set_params(**oversampling)
    X_train, y_train = ovs_model.fit_sample(df, y)

fileModel = model.fit(X_train, y_train)

save_params = {'base_model': fileModel, 'imputer': imputer}

joblib.dump(save_params,
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "models", "model_lgbm.pkl"))
