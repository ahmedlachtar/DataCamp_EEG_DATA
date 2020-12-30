import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

# import only when using original zip folders
# from prepare_data import _read_zip, private_public_split

problem_title = 'Classification of digit seen based on EEG signals'
_target_column_name = 'code'
_ignore_column_names = []
_prediction_label_names = [float(i) for i in range(10)]
_prediction_label_names.append(-1.0)

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


# READ DATA
devices = ['mu']  # 'in', 'mu', 'ep'


def string_to_float(string_array):
    return list(map(float, string_array.split(',')))


def _read_data(path, file_list):

    frames = []
    for file in file_list:
        df = pd.read_csv(os.path.join(path, 'data', 'public', file))
        frames.append(df)

    X = pd.concat(frames)
    X = X.sample(frac=1).reset_index(drop=True)
    X['data'] = X['data'].apply(string_to_float)
    y_array = X[_target_column_name].values.astype(int)
    X.drop(_target_column_name, inplace=True, axis=1)

    return X, y_array


def get_train_data(path='.'):
    file_list = []
    for device in devices:
        file_list.append(device + '_public_train.csv')

    return _read_data(path, file_list)


def get_test_data(path='.'):
    file_list = []
    for device in devices:
        file_list.append(device + '_public_test.csv')

    return _read_data(path, file_list)
