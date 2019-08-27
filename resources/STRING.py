import os
import datetime

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PATH
doc_input_path = root_path + '/data_input/'
doc_output_path = root_path + '/data_output/'
aux_path = root_path + '/data_extra/'
monitoring_path = root_path + '/monitoring/'
model_output_path = root_path + '/model_output/'

# FILES
test_identity = doc_input_path + 'test_identity.csv'
test_transaction = doc_input_path + 'test_transaction.csv'
train_identity = doc_input_path + 'train_identity.csv'
train_transaction = doc_input_path + 'train_transaction.csv'

output_train = doc_output_path + 'train.csv'
output_test = doc_output_path + 'test.csv'

monitoring_performance = monitoring_path + "model_performance.csv"
output_prediction = model_output_path + 'submission_' + str(datetime.datetime.today()) + '.csv'

kaggle_api = {"username": "sebastianmpalacio", "key": "33f016c1b26ab41531d46d6ec4e5607b"}

