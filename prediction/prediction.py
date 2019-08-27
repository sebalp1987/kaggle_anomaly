import os
import pandas as pd
import numpy as np

from sklearn.externals import joblib

from config.config import gral_parameters
from resources import STRING, df_utils

test = pd.read_csv(STRING.output_test, sep=',', encoding='utf-8')

dict_model = joblib.load(
        os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "models", "model_lgbm.pkl"))

# Preprocessing
X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
test = test[["TransactionDT", 'TransactionID']]

X_test = df_utils.clean_inf_nan(X_test)
imputer = dict_model.get('imputer')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns.values.tolist())

threshold_models = gral_parameters.get("threshold_models")
model = dict_model.get('base_model')

# Prediction
prediction = model.predcit_proba(X_test)
prediction = np.delete(prediction, 0, axis=1)


# Post-processing
prediction = pd.DataFrame(prediction, columns=['isFraud'], index=test.index)
prediction = np.where(prediction['isFraud'] > threshold_models, 1, 0)
prediction = pd.concat([test['TransactionID'], prediction], axis=1)
prediction.to_csv(STRING.output_prediction, index=False, sep=',')
